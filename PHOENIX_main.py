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

class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 100
    self.save_path = os.path.join('/work3/s204138/bach-models', 'PHOENIX_trained_models')
    self.default_checkpoint = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc')
    self.checkpoint_path = None #'/work3/s204138/bach-models/PHOENIX_trained_models/'  # if None train from scratch
    self.VOCAB_SIZE = 1085
    self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')
    self.batch_size = 2
    self.lr = 1e-4
    self.weight_decay = 1e-3
    self.num_workers = 0 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation #
    self.crop_size = 224
    # device #
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    ### initialize configs and device ###
    visual_encoder_CFG = visual_encoder_cfg()
    dp = DataPaths()
    CFG = cfg()
    torch.backends.cudnn.deterministic = True
    
    ### initialize data ###
    train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')[:2]
    val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')[:2]
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
    model = VisualEncoder(visual_encoder_CFG).to(CFG.device)
    
    ### train the model ###
    train(model, dataloader_train, dataloader_val, CFG)


if __name__ == '__main__':
  main()
