import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from s3d_backbone import VisualEncoder
from torch.utils.data import DataLoader

class cfg:
    def __init__(self):
        self.start_epoch = 0
        self.n_epochs = 80
        self.init_lr = -1000 # TODO look up value
        self.momentum = -1000 # TODO look up value
        self.epsilon = -1000 


class PhoenixPaths:
    def __init__(self):
        self.phoenix_labels = 'path/phoenix_labels.csv'
        self.phoenix_videos = 'path/phoenix_videos'

def main():
    CFG = cfg() 
    dp = PhoenixPaths()
    df = pd.read_csv(dp.phoenix_labels)
    vocab_size = len(set(df['sentences']))
    #### define dataloaders #####
    
    criterion = nn.CTCLoss(blank=vocab_size) # TODO make sure blank is not part of labels

    
def train(model, optimizer, dataloader, criterion):
    model.train()
    start = time.time()

    for i, (trg, ipt) in enumerate(dataloader):
        ipt = ipt.cuda()
        trg = trg.cuda()
        