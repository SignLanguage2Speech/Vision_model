import os
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_datasets.WLASLDataset import WLASLDataset
from models.S3D.model import S3D
from utils.load_weigths import load_kinetics_weights

import pdb

#import subprocess
#subprocess.call('pip list')
### opencv-python is here but cv2 gives importerror...

# stores directory paths for data on HPC
class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL_labels.csv"

class DataPaths_dummy:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL100"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL100_labels.csv"

class DataPathsWLASL1000:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL1000_labels.csv"
    

class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 100
    self.use_block = 5
    self.save_path = os.path.join('/work3/s204138/bach-models', 'trained_models')
    self.load_path = '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc' # ! Fill empty string with model file name
    self.checkpoint = "See load path" # start from scratch, i.e. epoch 0
    # self.checkpoint = True # start from checkpoint set in self.load_path
    self.batch_size = 6 # per GPU (they have 56 haha)
    self.lr = 0.1
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_workers = 4 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224
    self.seq_len = 64
    #self.epsilon = 1e-2 # TODO evaluate value 
    self.top_k = 10
    
    
def main():
  dp = DataPaths() # DATA PATHS
  #dp = DataPathsWLASL1000()
  #dp = DataPaths_dummy()
  CFG = cfg()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ############## load data ##############
  df = pd.read_csv(dp.wlasl_labels)
  img_folder = dp.wlasl_videos

  # get datasets
  WLASLtrain = WLASLDataset(df.loc[df['split']=='train'], img_folder, seq_len=CFG.seq_len,train=True, grayscale=False)
  WLASLval = WLASLDataset(df.loc[df['split']=='val'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
  WLASLtest = WLASLDataset(df.loc[df['split']=='test'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
  print("TRAIN LEN: ", len(df.loc[df['split']=='train']))

  ############## initialize model and optimizer ##############
  n_classes = len(set(df['gloss'])) #2000
  model = S3D(CFG.use_block).to(device)
  optimizer = optim.SGD(model.parameters(),
                        CFG.lr,
                        weight_decay=CFG.weight_decay,
                        momentum=CFG.momentum)
  
  ############## Load weights ##############
  if CFG.checkpoint is None: # if no newer model, use basics => pretrained on kinetics
    load_kinetics_weights(model, 'S3D_kinetics400.pt')
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
  
  else: # resume training
    model, optimizer, latest_epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(CFG.load_path, model, optimizer)
    CFG.start_epoch = latest_epoch

  ############## initialize dataloader ##############
  if CFG.multipleGPUs:
    dataloaderTrain = DataLoader(WLASLtrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers,
                                   pin_memory=True)
    model = torch.nn.parallel.DistributedDataParallel(model)
  
  else:
    print(f"Device: {device}")
    dataloaderTrain = DataLoader(WLASLtrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    dataloaderVal = DataLoader(WLASLval, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(WLASLtest, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    print(f"Model is on device: {device}")
    model.to(device)                   
  
  criterion = nn.CrossEntropyLoss().cuda()

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

  print("Starting training loop")
  for epoch in range(CFG.start_epoch, CFG.n_epochs):
    print(f"Epoch {epoch}")
    
    # run train loop
    #train_loss, train_acc = train(model, dataloaderTrain, optimizer, criterion, CFG)
    #train_losses.append(train_loss)
    #train_accs.append(train_acc)

    # run validation loop
    val_loss, val_acc = validate(model, dataloaderVal, criterion, CFG)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # update lr scheduler
    scheduler.step(np.mean(val_losses[-1]))

    ### Save checkpoint ###
    fname = os.path.join(CFG.save_path, f'S3D_WLASL-{epoch+1}_epochs-{np.mean(val_loss):.6f}_loss_{val_acc:5f}_acc')
    save_checkpoint(fname, model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs)
    
    
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
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nCurrent accuracy: {acc / (CFG.batch_size*(i+1)):.4f}\nTime: {(end - start)/60:.4f} min")

  acc = acc/len(dataloader.dataset)
  print(f"Final training accuracy: {acc}")
  return losses, acc

def validate(model, dataloader, criterion, CFG):
  losses = []
  model.eval()
  start = time.time()
  acc_top1 = 0
  acc_top5 = 0
  acc_top10 = 0
  print("################## Starting validation ##################")
  for i, (ipt, trg) in enumerate(dataloader):
    with torch.no_grad():
      ipt = ipt.cuda()
      trg = trg.cuda()
      #print(f"Input size: {ipt.size()}")
      #print("IPTT", ipt.size())
      #print("TRGG", trg.size())
      out = model(ipt)
      #print("OUTT", out.size())
      loss = criterion(out, trg)
      losses.append(loss.cpu())
      _, preds = out.topk(CFG.top_k, 1, True, True)
      preds = preds.squeeze(0)
      
      for j in range(len(preds)):
        if preds[j].item() == np.where(trg.cpu()[0] == 1)[0][0] and j == 0:
          acc_top1 += 1
          acc_top5 += 1
          acc_top10 += 1
          break
        elif preds[j].item() == np.where(trg.cpu()[0] == 1)[0][0] and j <= 4:
          acc_top5 += 1
          acc_top10 += 1
          break

        elif preds[j].item() == np.where(trg.cpu()[0] == 1)[0][0] and j <=9:
          acc_top10 += 1

          break
      
      end = time.time()
      if i % (CFG.print_freq/2) == 0:
        print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nTop-1 accuracy: {acc_top1 /(i+1):.4f}\nTop-5 accuracy: {acc_top5 /(i+1):.4f}\nTime: {(end - start)/60:.4f} min")

  acc_top1 = acc_top1/len(dataloader.dataset)
  acc_top5 = acc_top5/len(dataloader.dataset)
  acc_top10 = acc_top10/len(dataloader.dataset)
  print(f"Final top1 accuracy: {acc_top1}")
  print(f"Final top5 accuracy: {acc_top5}")
  print(f"Final top10 accuracy: {acc_top10}")
  return losses, acc_top1


def save_checkpoint(path, model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs):
  # save a general checkpoint
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'train_losses' : train_losses,
              'val_losses' : val_losses,
              'train_accs' : train_accs,
              'val_accs' : val_accs
              }, path)

def load_checkpoint(path, model, optimizer):
  print("### Loading model from checkpoint ###")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  train_losses = checkpoint['train_losses']
  val_losses = checkpoint['val_losses']
  train_accs = checkpoint['train_accs']
  val_accs = checkpoint['val_accs']
  return model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs


if __name__ == '__main__':
  main()

  
  
