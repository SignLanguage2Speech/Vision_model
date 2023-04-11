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
from datasets.PHOENIXDataset import PhoenixDataset, collator
from models.S3D_backbone import VisualEncoder
from datasets.preprocess_PHOENIX import getVocab, preprocess_df
from utils.get_train_loaders import get_train_loaders
from utils.load_checkpoint import load_checkpoint

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
    self.batch_size = 4
    self.lr = 1e-3
    self.weight_decay = 1e-3
    self.scheduler_reset_freq = 5
    self.num_workers = 8 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224


def main():

    ### initialize configs and device ###
    dp = DataPaths()
    CFG = cfg()
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### initialize data ###
    train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    ### initialize data ###
    PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train')
    PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='dev')
    PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='test')

    ### get dataloaders ###
    dataloader_train = DataLoader(PhoenixTrain, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    dataloader_val = DataLoader(PhoenixVal, batch_size=1, shuffle=False,num_workers=CFG.num_workers)
    dataloader_test = DataLoader(PhoenixTest, batch_size=1, shuffle=False, num_workers=CFG.num_workers) # TODO actually use this 🤡

    ### initialize model ###
    model = VisualEncoder(CFG.VOCAB_SIZE + 1).to(device)
    
    ### train the model ###
    train(model, dataloader_train, dataloader_val, CFG)


if __name__ == '__main__':
  main()
