import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import os
from torchaudio.models.decoder import ctc_decoder
from torchmetrics.functional import word_error_rate
from math import ceil

from torch.utils.data import Sampler, BatchSampler, SubsetRandomSampler
from random import shuffle

from PHOENIX.PHOENIXDataset import upsample, downsample
from PHOENIX.PHOENIXDataset import collator

from typing import Dict, Any

import pdb

torch.backends.cudnn.deterministic = True # CTC stuff

from PHOENIX.PHOENIXDataset import PhoenixDataset
from PHOENIX.preprocess_PHOENIX import getVocab, preprocess_df
from load_wlasl_weights import load_model_weights
from PHOENIX.s3d_backbone import VisualEncoder

class cfg:
    def __init__(self):
        self.start_epoch = 0
        self.n_epochs = 80
        self.init_lr = 1e-3
        self.weight_decay = 1e-3
        self.batch_size = 6
        self.num_workers = 8
        self.seq_len = 128
        self.VOCAB_SIZE = 1085
        # self.momentum = -1000 # TODO look up value
        self.epsilon = -1000  # TODO look up value
        self.model_path = '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc'
        # self.model_path = None
        self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')

class PhoenixPaths:
    def __init__(self):
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

class DynamicSampler(Sampler):
    def __init__(self, subset_dfs, batch_sizes, num_replicas=None, rank=None, seed=0, drop_last=False):
        self.total_samples = None # length of entire dataset
        self.num_samples = 0          # length of specific sampler's dataset
        self.batch_samplers = []
        for i,df in enumerate(subset_dfs):
            self.num_samples += ceil(len(df) / batch_sizes[i])
            subset_batch_sampler = BatchSampler(SubsetRandomSampler(list(df['index'])), batch_size=batch_sizes[i], drop_last=drop_last) # ! assumes dataframes retains original index col
            self.batch_samplers.append(subset_batch_sampler)
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        if num_replicas is None or rank is None: # if not parallel, terminate constructor here
            return

        self.total_samples = self.num_samples
        self.num_samples =  ceil(self.num_samples / self.num_replicas)

    def __iter__(self):
        all_batches = []
        shuffle(self.batch_samplers)
        for sampler in self.batch_samplers:
            for b in sampler:
                all_batches.append(b)
        if self.num_replicas is not None: # subsample batches
            all_batches = all_batches[self.rank:self.total_samples:self.num_replicas]
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        return self.num_samples # ? num_samples or total_samples?
    
    def set_epoch(self, epoch):
        print('setting epoch')
        self.epoch = epoch
    

class VisualEncoder_lightning(pl.LightningModule):
    def __init__(self, load_model=False):
        super().__init__()
        self.cfg = cfg()
        self.dp = PhoenixPaths()
        df = pd.read_csv(os.path.join(self.dp.phoenix_labels, f'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
        glosses = set([word for sent in df['orth'] for word in sent.split(' ')])
        self.vocab_size = len(glosses)
        self.model = VisualEncoder(self.vocab_size+1)     # ? number of classes is vocab_size + the blank character
        load_model_weights(self.model, self.cfg.model_path) if self.cfg.model_path is not None else self.model # initialize model with pretrained S3D backbone

        self.criterion = nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)  # ? blank character is indexed as first

        # self.automatic_optimization = False # ! For debugging

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_num):
        x,(y,target_lengths) = batch
        # x, ipt_lengths, y,target_lengths = batch
        out = self.model(x)
        log_probs = out.permute(1,0,2)
        input_lengths = torch.full(size=(out.shape[0],), fill_value=out.shape[1], dtype=torch.long)
        target_lengths = target_lengths.type(torch.long)#.cpu()

        refs = [t[:target_lengths[i]] for i, t in enumerate(y)]
        y = torch.concat(refs) # CuDNN - non-zero padded concatenated format

        loss = self.criterion(log_probs=log_probs, targets=y, input_lengths=(input_lengths), target_lengths=(target_lengths))

        return {'loss': loss}
    
    def validation_step(self, batch, batch_num):
        x,(y,target_lengths) = batch
        # x, ipt_lengths, y, target_lengths = batch
        out = self.model(x)
        log_probs = out.view(out.shape[1],out.shape[0],out.shape[2])
        input_lengths = torch.full(size=(out.shape[0],), fill_value=out.shape[1], dtype=torch.long)
        target_lengths = target_lengths.type(torch.long)#.cpu()

        refs = [t[:target_lengths[i]] for i, t in enumerate(y)]
        y = torch.concat(refs) # CuDNN - non-zero padded concatenated format

        loss = self.criterion(log_probs=log_probs, targets=y, input_lengths=(input_lengths), target_lengths=(target_lengths))

        CTC_decoder = ctc_decoder(
            lexicon=None, lm_dict=None, lm=None, # not using language model
            tokens= ['-'] + [str(i+1) for i in range(self.cfg.VOCAB_SIZE)] + ['|'], # vocab + blank and split
            nbest=1,             # number of hypotheses to return
            beam_size=100,       # n.o competing hypotheses at each step
            beam_size_token=25)  # top_n tokens to consider at each step
        out_decoder = CTC_decoder(torch.exp(out).cpu())
        WER = -1
        try:
            preds = [p[0].tokens for p in out_decoder]
            pred_sents = [TokensToSent(self.cfg.gloss_vocab, s) for s in preds] # predicted sentences
            ref_sents = [TokensToSent(self.cfg.gloss_vocab, s) for s in refs]   # reference sentences
            WER = word_error_rate(pred_sents, ref_sents).item()
            print()
            print_ref =  lambda idx: print(f'Actual    [{idx}]:' + str(ref_sents[idx]))
            print_pred = lambda idx: print(f'Predicted [{idx}]:' + str(pred_sents[idx]))
            for i in range(len(pred_sents)):
                print_ref(i)
                if str(pred_sents[i]) != '':
                    print_pred(i)
            print('WER:', WER)
        except IndexError:
            print(f"The output of the decoder:\n{out_decoder}\n caused an IndexError!")  
        
        return {'loss': loss, 'WER': WER}


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                        self.cfg.init_lr,
                        weight_decay=self.cfg.weight_decay)
        # return optimizer
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*1183*6*self.cfg.n_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5*1183*self.cfg.n_epochs)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "loss" # could be val loss
                }
            }

    def up_down_sample_helper(self, image_seq, max_len):
        img_seq_len = image_seq.size(1)
        if img_seq_len < max_len : 
            image_seq = upsample(image_seq, max_len)
        elif img_seq_len > max_len: # ? somehow 🤔
            image_seq = downsample(image_seq, max_len)
        return image_seq

    def collate_fn(self, batch):
        # print(batch)
        max_len, max_trg_len = 0, 0
        for img_seq, (trg, trg_len) in batch:
            max_len = max(img_seq.size(1), max_len)
            max_trg_len = max(trg_len, max_trg_len)
        # max_len = min(max_len,128) # truncate to ensure sufficient GPU memory (for batch size 6 on A100 40GB)
        # print("Max len of batch", max_len)
        img_seq_tensor = torch.zeros(len(batch), 3, max_len, 224, 224)
        trg_tensor = torch.zeros(len(batch),max_trg_len)
        trg_len_tensor = torch.zeros(len(batch))
        i = 0 # enumerate seems to be slower for some reason
        for (img_seq, (trg, trg_len)) in batch:
            img_seq_tensor[i] = self.up_down_sample_helper(img_seq, max_len)
            pad = torch.nn.ConstantPad1d((0, max_trg_len-len(trg)), value=-1)
            trg_tensor[i] = pad(trg)
            trg_len_tensor[i] = trg_len
            i += 1
        return img_seq_tensor, (trg_tensor, trg_len_tensor)

    def get_dataloader(self, split_type) -> DataLoader:
        df = pd.read_csv(os.path.join(self.dp.phoenix_labels, f'PHOENIX-2014-T.{split_type}.corpus.csv'), delimiter = '|')
        dataset = PhoenixDataset(df, 
                                 self.dp.phoenix_videos, 
                                 self.vocab_size, 
                                #  seq_len=self.cfg.seq_len, 
                                 split=split_type)
        if split_type == 'train':
            return DataLoader(dataset,
                          num_workers=self.cfg.num_workers, 
                          shuffle=False,
                          batch_sampler=DynamicSampler(*getSubsets(df)),#, num_replicas=2, rank=self.global_rank),
                        #   batch_sampler=DynamicSampler(*getSubsets(df), num_replicas=2, rank=torch._utils._get_device_index()),
                          collate_fn=self.collate_fn)
        else:
            return DataLoader(dataset, 
                          batch_size=1, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=False,
                          collate_fn=self.collate_fn)
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('dev')
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')

def TokensToSent(vocab, tokens):
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)

def getSubsets(train_df):
  dataframes = preprocess_df(train_df, split='train', save=False)
  
  # define batch sizes for datasets to avoid OOM on GPU (A100 40GB)
  batch_sizes = [8, 8, 8, 8, 8, 8, 8,
                 8, 8, 8, 6, 4, 4, 2]
  # V100 32GB
#   batch_sizes = [6, 6, 6, 6, 6, 6, 6, 
#                  6, 6, 6, 4, 2, 2, 2]

  return dataframes, batch_sizes