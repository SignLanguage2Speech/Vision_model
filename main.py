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
  WLASL = WLASLDataset(df, img_folder, seq_len=CFG.seq_len, grayscale=False)
  
  ############## load model ##############
  n_classes = len(set(df['gloss'])) #2000
  model = S3D(n_classes)
  if CFG.checkpoint is None:
    model = load_model_weights(model, 'S3D_kinetics400.pt')

  ############## initialize dataloader ##############
  if CFG.multipleGPUs:
    dataloader = DataLoader(WLASL, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers,
                                   pin_memory=True)
    model = torch.nn.parallel.DistributedDataParallel(model)
  
  else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    dataloader = DataLoader(WLASL, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    model.cuda()                   
      
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = optim.SGD(model.parameters(),
                        CFG.lr,
                        weight_decay=CFG.weight_decay,
                        momentum=CFG.momentum)
  
  for epoch in range(CFG.start_epoch, CFG.n_epochs):
    # adjust learning rate
    #adjust_lr(optimizer, CFG.lr_step, epoch)
    # run train loop
    train(model, dataloader, optimizer, criterion, CFG)
  

def train(model, dataloader, optimizer, criterion, CFG):
  losses = []
  model.train()
  start = time.time()
  for i, (ipt, trg) in enumerate(dataloader):
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

    if i % CFG.print_freq == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses)}\nTime: {np.round(start, 2)/60} min")


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
  
#s3d.pytorch