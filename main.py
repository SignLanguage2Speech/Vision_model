import os
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.WLASLDataset import WLASLDataset
from model import S3D
from utils.load_weigths import load_model_weights


#import subprocess
#subprocess.call('pip list')
### opencv-python is here but cv2 gives importerror...


class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 80
    self.save_path = os.path.join(os.getcwd(), 'trained_models')
    self.load_path = os.path.join(self.save_path, 'SOME_MODEL_NAME')
    self.checkpoint = None
    self.batch_size = 6 # per GPU (they have 56 haha)
    self.lr = 0.1
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_workers = 4
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size=224
    self.seq_len = 64
    
    
def main():
  CFG = cfg()
  ############## load data ##############
  df = pd.read_csv('data/WLASL/WLASL_labels.csv')
  img_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_videos')

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
  if CFG.checkpoint is None:
    model = load_model_weights(model, 'S3D_kinetics400.pt')
    train_losses = []
    val_losses = []

  else: # resume training
    model, optimizer, latest_epoch, train_losses, val_losses = load_checkpoint(CFG.load_path, model, optimizer)
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
    dataloaderVal = DataLoader(WLASLtest, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    model.cuda()                   
      
  criterion = nn.CrossEntropyLoss().cuda()
  
  train_losses = []
  val_losses = []

  for epoch in range(CFG.start_epoch, CFG.n_epochs):
    # adjust learning rate
    if len(train_losses) > 0:
      if np.mean(train_loss) - np.mean(val_loss) < CFG.epsilon:
        adjust_lr(optimizer, CFG.lr_step, epoch)

    # run train loop
    train_loss = train(model, dataloaderTrain, optimizer, criterion, CFG)
    train_losses.append(train_loss)

    # run validation loop
    val_loss = validate(model, dataloaderVal, criterion, CFG)
    val_losses.append(val_loss)

    # save progress
    if epoch == CFG.start_epoch:
      fname = os.path.join(CFG.save_path, f'S3D_WLASL_{epoch}epoch_{np.round(np.mean(val_losses), 3)}_loss')
      save_checkpoint(fname, model, optimizer)

    # check if the current model has the lowest validation loss
    elif np.argmin(np.mean(val_losses, axis=1)) == len(val_losses) - 1:
      loss_rounded = np.round(np.mean(val_losses, axis=1)[-1], 3)
      fname = os.path.join(CFG.save_path, f'S3D_WLASL_{epoch}epochs_{loss_rounded}_loss')
      save_checkpoint(fname, model, optimizer)



def train(model, dataloader, optimizer, criterion, CFG):
  losses = []
  model.train()
  start = time.time()

  for i, (ipt, trg) in enumerate(dataloader):
    print(f"processed images size: {ipt.size()}")
    ipt = ipt.squeeze(1).cuda()
    trg=trg.cuda()
    ipt_var = torch.autograd.Variable(ipt)
    trg_var = torch.autograd.Variable(trg)

    out = model(ipt_var)[0]
    preds = nn.Softmax(out)
    loss = criterion(preds, trg_var)
    losses.append(loss)

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
    
    trg = trg.cuda()
    ipt_var = torch.autograd.Variable(ipt)
    trg_var = torch.autograd.Variable(trg)

    out = model(ipt_var)[0]
    preds = nn.Softmax(out)
    loss = criterion(preds, trg_var)
    losses.append(loss)
    
    end = time.time()
    if i % CFG.print_freq == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses)}\nTime: {np.round(end - start, 2)/60} min")

  return losses


def adjust_lr(optimizer, CFG):
  decay_factor = 10
  for param in optimizer.param_groups:
    param['lr'] = param['lr'] / decay_factor
    param['weight_decay'] = CFG.weight_decay


def save_checkpoint(path, model, optimizer, epoch, train_losses, val_losses, CFG):
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
  main()
  """
  seq_len = 64
  initial = torch.zeros(1, 3, 30, 333, 333)

  imgs = os.listdir(os.path.join(os.getcwd(), 'data/WLASL/WLASL_images'))
  os.chdir(os.path.join(os.getcwd(), 'data/WLASL/WLASL_images'))
  lengths = [len(os.listdir(f)) for f in imgs]
  #start_idx = np.random.randint(0, seq_len-initial.size(2))
  print(f"min: {min(lengths)} idx: {np.argmin(lengths)}\nmax: {max(lengths)}")
  """