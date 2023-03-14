import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import os

import pdb

torch.backends.cudnn.deterministic = True

from PHOENIX.PHOENIXDataset import PhoenixDataset
from load_wlasl_weights import load_model_weights
from visual_encoder import VisualEncoder

class cfg:
    def __init__(self):
        self.start_epoch = 0
        self.n_epochs = 80
        self.init_lr = 1e-3
        self.weight_decay = 1e-3
        self.batch_size = 2
        self.seq_len = 128
        # self.momentum = -1000 # TODO look up value
        self.epsilon = -1000  # TODO look up value
        self.model_path = '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc'
        # self.model_path = None

class PhoenixPaths:
    def __init__(self):
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

class VisualEncoder_lightning(pl.LightningModule):
    def __init__(self, load_model=False):
        super().__init__()
        self.cfg = cfg()
        self.dp = PhoenixPaths()
        df = pd.read_csv(os.path.join(self.dp.phoenix_labels, f'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
        glosses = set([word for sent in df['orth'] for word in sent.split(' ')])
        self.vocab_size = len(glosses)
        self.model = VisualEncoder(self.vocab_size+1)     # ? number of classes is vocab_size + the blank character

        # initialize model with pretrained S3D backbone
        self.model = load_model_weights(self.model, self.cfg.model_path) if self.cfg.model_path is not None else self.model

        # self.criterion = nn.CTCLoss(blank=0)  # ? blank character is indexed as last
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # ? blank character is indexed as last
        # self.criterion = nn.CTCLoss(blank=self.vocab_size, reduction='sum', zero_infinity=True)  # ? blank character is indexed as last

        self.optimizer = None # TODO implement loading from checkpoint

        # self.automatic_optimization = False # ! For debugging

    def forward(self, x):
        return self.model(x)

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        # pdb.set_trace()
        print(loss)
        loss.backward()

    def training_step(self, batch, batch_num):
        x,(y,target_lengths) = batch
        # out = torch.log(self.model(x))
        out = self.model(x)
        log_probs = out.view(out.shape[1],out.shape[0],out.shape[2])
        input_lengths = torch.full(size=(out.shape[0],), fill_value=out.shape[1], dtype=torch.long)
        target_lengths = target_lengths.type(torch.long).cpu()

        # y = y.reshape(-1).cpu() # CuDNN - zero padded concatenated format
        y = torch.concat([q[:target_lengths[i]] for i,q in enumerate(y)]).cpu() # CuDNN - non-zero padded concatenated format

        # opt = self.optimizers()
        # opt.zero_grad()
        # pdb.set_trace()
        loss = self.criterion(log_probs=log_probs, targets=y, input_lengths=(input_lengths), target_lengths=(target_lengths))
        # self.manual_backward(loss)
        # opt.step()

        # print(loss)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return self.optimizer if self.optimizer is not None else \
            optim.Adam(self.model.parameters(),
                    self.cfg.init_lr,
                    weight_decay=self.cfg.weight_decay)

    def get_dataloader(self, split_type) -> DataLoader:
        df = pd.read_csv(os.path.join(self.dp.phoenix_labels, f'PHOENIX-2014-T.{split_type}.corpus.csv'), delimiter = '|')
        dataset = PhoenixDataset(df, self.dp.phoenix_videos, self.vocab_size, seq_len=self.cfg.seq_len, split=split_type)
        return DataLoader(dataset, batch_size=self.cfg.batch_size if split_type=='train' else 1, num_workers=4)
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('dev')
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')