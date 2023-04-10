import os
import numpy as np
import pandas as pd
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchaudio.models.decoder import ctc_decoder
from torchmetrics.functional import word_error_rate

from utils.load_weigths import load_PHOENIX_weights
from datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from models.S3D_backbone import VisualEncoder
from datasets.preprocess_PHOENIX import getVocab, preprocess_df

import pdb

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 100
    self.save_path = os.path.join('/work3/s204138/bach-models', 'PHOENIX_trained_models')
    self.default_checkpoint = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc')
    self.checkpoint_path = None #'/work3/s204138/bach-models/PHOENIX_trained_models/'  # if None train from scratch
    #self.checkpoint_path = None
    self.VOCAB_SIZE = 1085
    self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')
    self.batch_size = None # per GPU (they have 56 haha)
    self.lr = 1e-3
    self.weight_decay = 1e-3
    self.scheduler_reset_freq = 5
    self.num_workers = 8 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224


def main():
    print("################## Running main ##################")
    dp = DataPaths()
    CFG = cfg()
    torch.backends.cudnn.deterministic = True
    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ################## Load and prepare data ##################
    train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    train_augmentations = DataAugmentations(split_type='train')
    dataloadersTrain = getTrainLoaders(train_df, lambda data: collator(data, train_augmentations), dp, CFG)
    PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='dev')
    PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='test')
    
    val_augmentations = DataAugmentations()
    dataloaderVal = DataLoader(PhoenixVal, batch_size=1, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers,
                                   collate_fn=lambda data: collator(data, train_augmentations))
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(PhoenixTest, batch_size=1, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers,
                                   collate_fn=lambda data: collator(data, train_augmentations))

    ################## Initialization ##################
    model = VisualEncoder(CFG.VOCAB_SIZE + 1).to(device)
    optimizer = optim.AdamW(model.parameters(),
                          lr = CFG.lr,
                          weight_decay = CFG.weight_decay)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_df)/CFG.batch_size)*5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(train_df)/CFG.batch_size)*CFG.scheduler_reset_freq)

    if CFG.checkpoint_path is not None:
      print("################## Loading checkpointed weights ##################")
      model, optimizer, scheduler, current_epoch, train_losses, val_losses, train_WERS, val_WERS = load_checkpoint(CFG.checkpoint_path, model, optimizer, scheduler)
      CFG.start_epoch = current_epoch
    
    else:
      print("################## Loading WLASL weights ##################")
      load_PHOENIX_weights(model, CFG.default_checkpoint, verbose=True)
      # initialize optimizer again with updated params
      optimizer = optim.AdamW(model.parameters(),
                          lr = CFG.lr,
                          weight_decay = CFG.weight_decay)
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(train_df)/CFG.batch_size) * CFG.scheduler_reset_freq)
      train_losses = []
      train_WERS = []
      val_losses = []
      val_WERS = []
      
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False, reduction='mean').to(device) # zero_infinity is for debugging purposes only...

    print(f"Model is on device: {device}")
    model.to(device)  

    CTC_decoder = ctc_decoder(
    lexicon=None,       # not using language model
    lm_dict=None,       # not using language model
    lm=None,            # not using language model
    tokens= ['-'] + [str(i+1) for i in range(CFG.VOCAB_SIZE)] + ['|'], # vocab + blank and split
    nbest=1, # number of hypotheses to return
    beam_size = 100,       # n.o competing hypotheses at each step
    beam_size_token=25,  # top_n tokens to consider at each step
    )

    for i in range(CFG.start_epoch, CFG.n_epochs):
      print(f"Current epoch: {i+1}")
      # run train loop
      train_loss, train_WER = train(model, dataloadersTrain, optimizer, criterion, scheduler, CTC_decoder, CFG)
      train_losses.append(train_loss)
      train_WERS.append(train_WER)

      # run validation loop
      val_loss, val_WER = validate(model, dataloaderVal, criterion, CTC_decoder, CFG)
      val_losses.append(val_loss)
      val_WERS.append(val_WER)

      ### Save checkpoint ###
      fname = os.path.join(CFG.save_path, f'S3D_PHOENIX-{i+1}_epochs-{np.mean(val_loss):.6f}_loss_{np.mean(val_WER):5f}_WER')
      save_checkpoint(fname, model, optimizer, scheduler, i+1, train_losses, val_losses, train_WERS, val_WERS)

def train(model, dataloaders, optimizer, criterion, scheduler, decoder, CFG):
  losses = []
  model.train()
  start = time.time()
  WERS = []

  print("################## Starting training ##################")
  random.shuffle(dataloaders) # shuffle dataloaders (inplace)
  for dataloader in dataloaders:
    print("DATASET LEN: ", len(dataloader.dataset))
    for i, (ipt, _, trg, trg_len) in enumerate(dataloader):
      ipt = ipt.cuda()
      trg = trg.cuda()

      refs = [t[:trg_len[i]].cpu() for i, t in enumerate(trg)]
      ref_sents = [TokensToSent(CFG.gloss_vocab, s) for s in refs]

      out=model(ipt)
      x = out.permute(1, 0, 2)
      trg = torch.concat([q[:trg_len[i]] for i,q in enumerate(trg)])
      trg_len = trg_len.to(torch.int32)
      ipt_len = torch.full(size=(out.size(0),), fill_value = out.size(1), dtype=torch.int32)

      with torch.backends.cudnn.flags(enabled=False):
        loss = criterion(x, 
                        trg,#.cpu(), 
                        input_lengths=ipt_len,
                        target_lengths=trg_len)
        
      losses.append(loss.detach().cpu().item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #print("LATEST LRR", scheduler.get_last_lr())
      scheduler.step()

      out_d = decoder(torch.exp(out).cpu())
      try:
        preds = [p[0].tokens for p in out_d]
        #print("PREDSSS", preds)
        pred_sents = [TokensToSent(CFG.gloss_vocab, s) for s in preds]
        WERS.append(word_error_rate(pred_sents, ref_sents).item())

      
      except IndexError:
        #pdb.set_trace()
        print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")
      
      end = time.time()
      if max(1, i) % (CFG.print_freq) == 0:

        print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nAvg WER: {np.mean(WERS):.4f}\nTime: {(end - start)/60:.4f} min")
        print("PREDICTIONS:\n", pred_sents)
        print("###Reference:\n", ref_sents)
      
      #if max(1, i) % 2 == 0:
        #print("###Prediction:\n", pred_sents)
        #print("###Reference:\n", ref_sents)
        #break

    print(f"Avg. WER after dataset: {np.mean(WERS)}")
  print(f"Final avg. WER: {np.mean(WERS)}")
  return losses, WERS

def validate(model, dataloader, criterion, decoder, CFG):
  losses = []
  model.eval()
  start = time.time()
  WERS = []
  print("################## Starting validation ##################")
  for i, (ipt, _, trg, trg_len) in enumerate(dataloader):
    with torch.no_grad():
      ipt = ipt.cuda()
      trg = trg.cuda()
      
      out = model(ipt)
      x = out.permute(1, 0, 2)
      ipt_len = torch.full(size=(1,), fill_value = out.size(1), dtype=torch.int32)

      #print("IPTT", ipt.size())
      #print("IPT LEN ", ipt_len)
      #print("OUTT", out.size())
      #print("TRGG", trg.size())
      #print("TRG LEN", trg_len)
      with torch.backends.cudnn.flags(enabled=False):
        loss = criterion(x, 
                      trg, 
                      input_lengths=ipt_len.cuda(),
                      target_lengths=trg_len.cuda())

      losses.append(loss.detach().cpu().item())
      
      out_d = decoder(torch.exp(out).cpu())
      try:
        preds = [p[0].tokens for p in out_d]
        pred_sents = [TokensToSent(CFG.gloss_vocab, s) for s in preds]
        ref_sents = TokensToSent(CFG.gloss_vocab, trg[0][:trg_len[0]])
        WERS.append(word_error_rate(pred_sents, ref_sents).item())
    
      except IndexError:
        print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")
        #pdb.set_trace()

      end = time.time()
      if max(1, i) % (CFG.print_freq/2) == 0:
        print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nAvg WER: {np.mean(WERS):.4f}\nTime: {(end - start)/60:.4f} min")
        print("PREDICTIONS:\n", pred_sents)
        print("REFERENCES:\n", ref_sents)
      
      #if max(1, i) % 5 == 0:
        #print("###Prediction:\n", pred_sents)
        #print("###Reference:\n", ref_sents)
        #break
     
  print(f"Final avg. WER val: {np.mean(WERS)}")
  return losses, WERS


def save_checkpoint(path, model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS):
  # save a general checkpoint
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'scheduler_state_dict' : scheduler.state_dict(),
              'train_losses' : train_losses,
              'val_losses' : val_losses,
              'train_WERS' : train_WERS,
              'val_WERS' : val_WERS
              }, path)

def load_checkpoint(path, model, optimizer, scheduler):
  print("### Loading model from checkpoint ###")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']
  train_losses = checkpoint['train_losses']
  val_losses = checkpoint['val_losses']
  train_WERS = checkpoint['train_WERS']
  val_WERS = checkpoint['val_WERS']
  return model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS

def TokensToSent(vocab, tokens):
  
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)

def getTrainLoaders(train_df, collator, dp, CFG):
  dataframes = preprocess_df(train_df, split='train', save=False)
  print("N datasets in train: ", len(dataframes))
  dataLoaders = []
  
  # define batch sizes for datasets to avoid OOM
  batch_sizes = [8, 8, 8, 8, 8, 8, 8,
                 8, 8, 8, 6, 4, 4, 2]
  for i, df in enumerate(dataframes):
    dataLoaders.append(DataLoader(PhoenixDataset(df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train'),
                                  batch_size=batch_sizes[i],
                                  shuffle=True, 
                                  num_workers=CFG.num_workers,
                                  collate_fn=collator))

  return dataLoaders

if __name__ == '__main__':
  main()
