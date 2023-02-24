import os
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.WLASLDataset import WLASLDataset
from model import S3D
from utils.load_weigths import load_model_weights


import pdb

#import subprocess
#subprocess.call('pip list')
### opencv-python is here but cv2 gives importerror...

# stores directory paths for data on HPC
class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204503/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204503/bach-data/WLASL/WLASL_labels.csv"

class DataPaths_dummy:
  def __init__(self):
    self.wlasl_videos = "/work3/s204503/bach-data/WLASL/WLASL100"
    self.wlasl_labels = "/work3/s204503/bach-data/WLASL/WLASL100_labels.csv"


class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 80
    self.save_path = os.path.join('/work3/s204503/bach-models', 'trained_models')
    self.load_path = os.path.join(self.save_path, 'S3D_WLASL_14epochs_2.4790000915527344_loss')
    self.checkpoint = None # start from scratch, i.e. epoch 0
    # self.checkpoint = True # start from checkpoint set in self.load_path
    self.batch_size = 6 # per GPU (they have 56 haha)
    self.lr = 0.1
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_workers = 4 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size=224
    self.seq_len = 64
    self.epsilon = 1e-2 # TODO evaluate value 
    
    
def main():
  # dp = DataPaths() # DATA PATHS
  dp = DataPaths_dummy()
  CFG = cfg()
  ############## load data ##############
  df = pd.read_csv(dp.wlasl_labels)
  img_folder = dp.wlasl_videos
  # WLASL = WLASLDataset(df, img_folder, seq_len=CFG.seq_len, grayscale=False)
  
  # Get datasets
  WLASLtrain = WLASLDataset(df.loc[df['split']=='train'], img_folder, seq_len=CFG.seq_len,train=True, grayscale=False)
  WLASLval = WLASLDataset(df.loc[df['split']=='val'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
  WLASLtest = WLASLDataset(df.loc[df['split']=='test'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
  print("TRAIN LEN: ", len(df.loc[df['split']=='train']))
  ############## initialize model and optimizer ##############
  n_classes = len(set(df['gloss'])) #2000
  model = S3D(n_classes)
  optimizer = optim.SGD(model.parameters(),
                        CFG.lr,
                        weight_decay=CFG.weight_decay,
                        momentum=CFG.momentum)

  ############## Load weights ##############
  if CFG.checkpoint is None: # if no newer model, use basics => pretrained on kinetics
    model = load_model_weights(model, 'S3D_kinetics400.pt')
    train_losses = []
    val_losses = []
  else: # resume training
    model, optimizer, latest_epoch, train_losses, val_losses = load_checkpoint(CFG.load_path, model, optimizer)
    #model.to('cuda') # TODO when is it cleanest to send to cuda? ### is this redundant? (see line 110)
    CFG.start_epoch = latest_epoch

  ############## initialize dataloader ##############
  if CFG.multipleGPUs:
    dataloaderTrain = DataLoader(WLASLtrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers,
                                   pin_memory=True)
    model = torch.nn.parallel.DistributedDataParallel(model)
  
  else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    dataloaderTrain = DataLoader(WLASLtrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    dataloaderVal = DataLoader(WLASLval, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(WLASLtest, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    print("Transferring model to GPU")
    model.to(device)                   
  
  criterion = nn.CrossEntropyLoss().cuda()

  print("Starting training loop")
  for epoch in range(CFG.start_epoch, CFG.n_epochs):
    train_loss = train_losses[-1]
    print(f"Epoch {epoch}")
    
    # run train loop
    train_loss = train(model, dataloaderTrain, optimizer, criterion, CFG)
    train_losses.append(train_loss)

    # run validation loop
    val_loss = validate(model, dataloaderVal, criterion, CFG)
    val_losses.append(val_loss)

    # adjust learning rate
    if len(train_losses) > 0:
      if np.abs(np.mean(train_loss) - np.mean(val_loss)) < CFG.epsilon:
        adjust_lr(optimizer, CFG)

    ### Save checkpoint ###
    # check if the current model has the lowest validation loss
    if np.argmin(np.mean(val_losses, axis=1)) == len(val_losses) - 1:
      loss_rounded = np.round(np.mean(val_losses, axis=1)[-1], 3)
      fname = os.path.join(CFG.save_path, f'S3D_WLASL_{epoch}epochs_{loss_rounded}_loss')
      save_checkpoint(fname, model, optimizer, epoch, train_losses, val_losses)
      # TODO Remove all previously saved models

def train(model, dataloader, optimizer, criterion, CFG):
  losses = []
  model.train()
  start = time.time()


  for i, (ipt, trg) in enumerate(dataloader):
    # pdb.set_trace()
    # print(f"processed images size: {ipt.size()}")
    ipt = ipt.cuda()
    trg = trg.cuda()
    ipt_var = torch.autograd.Variable(ipt)
    trg_var = torch.autograd.Variable(trg)

    out = model(ipt_var)
    # pdb.set_trace()
    preds = F.softmax(out,dim=1)
    loss = criterion(preds, trg_var)
    losses.append(loss.detach().cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    end = time.time()
    if i % CFG.print_freq == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses)}\nTime: {np.round(end - start, 2)/60} min")

  return losses

def validate(model, dataloader, criterion, CFG):
  losses = []
  model.eval()
  start = time.time()

  for i, (ipt, trg) in enumerate(dataloader):
    
    ipt = ipt.cuda()
    trg = trg.cuda()
    ipt_var = torch.autograd.Variable(ipt)
    trg_var = torch.autograd.Variable(trg)

    out = model(ipt_var)
    preds = F.softmax(out, dim=1)
    loss = criterion(preds, trg_var)
    losses.append(loss.detach().cpu())
    
    end = time.time()
    if i % CFG.print_freq == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses)}\nTime: {np.round(end - start, 2)/60} min")

  return losses


def adjust_lr(optimizer, CFG):
  decay_factor = 10
  for param in optimizer.param_groups:
    param['lr'] = param['lr'] / decay_factor
    param['weight_decay'] = CFG.weight_decay


def save_checkpoint(path, model, optimizer, epoch, train_losses, val_losses):
  # save a general checkpoint
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'train_losses' : train_losses,
              'val_losses' : val_losses
              }, path)


def load_checkpoint(path, model, optimizer):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  train_losses = checkpoint['train_losses']
  val_losses = checkpoint['val_losses']
  return model, optimizer, epoch, train_losses, val_losses


if __name__ == '__main__':
  # freeze_support()
  main()