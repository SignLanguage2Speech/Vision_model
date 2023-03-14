import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PHOENIX.PHOENIXDataset import PhoenixDataset
from PHOENIX.s3d_backbone import VisualEncoder



#annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
#features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 80
    self.save_path = os.path.join('/work3/s204138/bach-models', 'continued_WLASL_training')
    self.load_path = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc') # ! Fill empty string with model file name
    self.checkpoint = "See load path" # start from scratch, i.e. epoch 0
    self.VOCAB_SIZE = 1085
    # self.checkpoint = True # start from checkpoint set in self.load_path
    self.batch_size = 8 # per GPU (they have 56 haha)
    self.lr = 1e-3
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_workers = 4 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224
    self.seq_len = 128
    #self.epsilon = 1e-2 # TODO evaluate value 
    self.continue_lr = 0.005


def main():
    dp = DataPaths()
    CFG = cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

   
    PhoenixTrain = PhoenixDataset(train, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, train=True)
    PhoenixVal = PhoenixDataset(val, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, train=False)
    PhoenixTest = PhoenixDataset(test, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, train=False)

    model = VisualEncoder(CFG.VOCAB_SIZE + 1)
    optimizer = optim.Adam(model.parameters(),
                           lr = CFG.lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.n_epochs)


    criterion = torch.nn.CTCLoss(blank=CFG.VOCAB_SIZE)


    dataloaderTrain = DataLoader(PhoenixTrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    dataloaderVal = DataLoader(PhoenixVal, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(PhoenixTest, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    
    print(f"Model is on device: {device}")
    model.to(device)  
    train_losses, train_accs = [], []


    for i in range(CFG.start_epoch, CFG.n_epochs):
      print(f"Current epoch: {i+1}")
      # run train loop
      train_loss, train_acc = train(model, dataloaderTrain, optimizer, criterion, CFG)
      train_losses.append(train_loss)
      train_accs.append(train_acc)

def train(model, dataloader, optimizer, criterion, CFG):
  losses = []
  model.train()
  start = time.time()
  acc = 0
  print("################## Starting training ##################")
  for i, (ipt, trg) in enumerate(dataloader):

    ipt = ipt.cuda()
    trg = trg.cuda()

    out = model(ipt)
    loss = criterion(out, trg)
    losses.append(loss.detach().cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # compute model accuracy
    _, preds = out.topk(1, 1, True, True)
    for j in range(len(preds)):
      if preds[j] == np.where(trg.cpu()[j] == 1)[0][0]:
        acc += 1

    end = time.time()
    if i % (CFG.print_freq) == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nTime: {(end - start)/60:.4f} min")

  
  print(f"Final training accuracy: {acc}")
  return losses





