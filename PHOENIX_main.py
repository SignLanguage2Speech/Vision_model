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
from models.VisualEncoder import VisualEncoder
from train_datasets.preprocess_PHOENIX import getVocab, preprocess_df
from utils.load_checkpoint import load_checkpoint
from train_datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from train.trainer import train

from configs.VisualEncoderConfig import cfg as visual_encoder_cfg

import pdb

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

def main():

    ### initialize configs and device ###
    dp = DataPaths()
    CFG = visual_encoder_cfg()
    torch.backends.cudnn.deterministic = True
    
    ### initialize data ###
    train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    ### initialize data ###
    PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train')
    PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='dev')
    PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='test')

    ### get dataloaders ###
    train_augmentations = DataAugmentations(split_type='train')
    val_augmentations = DataAugmentations(split_type='val')
    dataloader_train = DataLoader(
      PhoenixTrain, 
      collate_fn = lambda data: collator(data, train_augmentations), 
      batch_size=CFG.batch_size, 
      shuffle=True, num_workers=CFG.num_workers)
    dataloader_val = DataLoader(
      PhoenixVal, 
      collate_fn = lambda data: collator(data, val_augmentations), 
      batch_size=1, 
      shuffle=False,
      num_workers=CFG.num_workers)
    dataloader_test = DataLoader(
      PhoenixTest, 
      collate_fn = lambda data: collator(data, val_augmentations), 
      batch_size=1, 
      shuffle=False, 
      num_workers=CFG.num_workers) # TODO actually use this 🤡
    
    ### initialize model ###
    model = VisualEncoder(CFG).to(CFG.device)
    
    ### train the model ###
    train(model, dataloader_train, dataloader_val, CFG)


if __name__ == '__main__':
  main()
