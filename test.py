from models.VisualEncoder import VisualEncoder
from models.S3D_backbone import S3D_backbone
import torch
import time, os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
#import tensorflow as tf

from configs.VisualEncoderConfig import cfg
from train.trainer import validate, get_train_modules
from train_datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from train_datasets.preprocess_PHOENIX import getVocab, preprocess_df

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

CFG = cfg()
dp = DataPaths()






### initialize data ###
train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
### initialize data ###
PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train')
PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='dev')
### get dataloaders ###
train_augmentations = DataAugmentations(split_type='val')
val_augmentations = train_augmentations # DataAugmentations(split_type='val')

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

### initialize model ###
model = VisualEncoder(CFG).to(CFG.device)

optimizer, criterion, scheduler, \
    decoder, train_losses, train_word_error_rates, \
    val_losses, val_word_error_rates = get_train_modules(model, dataloader_train, CFG)

### validate the model ###
validate(model, dataloader_val, criterion, decoder, CFG)
