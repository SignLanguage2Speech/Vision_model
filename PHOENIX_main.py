import os
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchaudio.models.decoder import ctc_decoder
from torchmetrics.functional import word_error_rate

from PHOENIX.PHOENIXDataset import PhoenixDataset
from PHOENIX.s3d_backbone import VisualEncoder
from PHOENIX.preprocess_PHOENIX import getVocab

import pdb

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

class cfg:
  def __init__(self):
    self.start_epoch = 0
    self.n_epochs = 80
    self.save_path = os.path.join('/work3/s204138/bach-models', 'PHOENIX_trained_models')
    self.load_path = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc') # ! Fill empty string with model file name
    self.checkpoint = "See load path" # start from scratch, i.e. epoch 0
    self.VOCAB_SIZE = 1085
    self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')
    # self.checkpoint = True # start from checkpoint set in self.load_path
    self.batch_size = 8 # per GPU (they have 56 haha)
    self.lr = 1e-3
    self.weight_decay = 1e-3
    self.num_workers = 8 # ! Set to 0 for debugging
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224
    self.seq_len = 128


def main():
    dp = DataPaths()
    CFG = cfg()
    torch.backends.cudnn.deterministic = True
    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, split='train')
    PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, split='dev')
    PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, seq_len=CFG.seq_len, split='test')

    model = VisualEncoder(CFG.VOCAB_SIZE + 1)
    optimizer = optim.Adam(model.parameters(),
                           lr = CFG.lr,
                           weight_decay = CFG.weight_decay)
    
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(train_df)/CFG.batch_size)*4)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(100/CFG.batch_size)*2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_df)/CFG.batch_size)*4)

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False).to(device) # zero_infinity is for debugging purposes only...

    dataloaderTrain = DataLoader(PhoenixTrain, batch_size=CFG.batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    dataloaderVal = DataLoader(PhoenixVal, batch_size=1, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers)
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(PhoenixTest, batch_size=1, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers)
    
    print(f"Model is on device: {device}")
    model.to(device)  

    CTC_decoder = ctc_decoder(
    lexicon=None,       # not using language model
    lm_dict=None,       # not using language model
    lm=None,            # not using language model
    tokens= ['-'] + [str(i+1) for i in range(CFG.VOCAB_SIZE)] + ['|'], # vocab + blank and split
    nbest=1, # number of hypotheses to return
    beam_size = 100,       # n.o competing hypotheses at each step
    beam_size_token=25,  # top_n tokens to consider at each step
    )

    train_losses = []
    train_WERS = []
    val_losses = []
    val_WERS = []

    for i in range(CFG.start_epoch, CFG.n_epochs):
      print(f"Current epoch: {i+1}")
      # run train loop
      train_loss, train_WER = train(model, dataloaderTrain, optimizer, criterion, scheduler, CTC_decoder, CFG)
      train_losses.append(train_loss)
      train_WERS.append(train_WER)

      # run validation loop
      val_loss, val_WER = validate(model, dataloaderVal, criterion, CTC_decoder, CFG)
      val_losses.append(val_loss)
      val_WERS.append(val_WER)

      ### Save checkpoint ###
      fname = os.path.join(CFG.save_path, f'S3D_PHOENIX-{i+1}_epochs-{np.mean(val_loss):.6f}_loss_{np.mean(val_WER):5f}_WER')
      save_checkpoint(fname, model, optimizer, scheduler, i+1, train_losses, val_losses, train_WERS, val_WERS)

def train(model, dataloader, optimizer, criterion, scheduler, decoder, CFG):
  losses = []
  model.train()
  start = time.time()
  WERS = []
  print("################## Starting training ##################")
  for i, (ipt, trg, trg_len) in enumerate(dataloader):

    ipt = ipt.cuda()
    trg = trg.cuda()
    ipt_len = torch.full(size=(CFG.batch_size,), fill_value=CFG.seq_len/4, dtype=torch.int32)
    
    refs = [t[:trg_len[i]].cpu() for i, t in enumerate(trg)]
    ref_sents = [TokensToSent(CFG.gloss_vocab, s) for s in refs]

    out=model(ipt)
    #out.view(out.size(2), out.size(0), out.size(1)) #### For convolution w. kernel size 1.
    x = out.view(out.size(1), out.size(0), out.size(2))
    trg = torch.concat([q[:trg_len[i]] for i,q in enumerate(trg)])
    trg_len = trg_len.to(torch.int32)

    with torch.backends.cudnn.flags(enabled=False):
      loss = criterion(x, 
                      trg,#.cpu(), 
                      input_lengths=ipt_len.cuda(),
                      target_lengths=trg_len.cuda())
      
    losses.append(loss.detach().cpu().item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print("LATEST LRR", scheduler.get_last_lr())
    scheduler.step()
    
    #out_d = decoder(torch.exp(out - 1e-16).cpu().view(out.size(0), out.size(2), out.size(1))) ### Used for 1d convolutions w kernel size 1
    out_d = decoder(torch.exp(out).cpu())

    try:
      preds = [p[0].tokens for p in out_d]
      #print("PREDSSS", preds)
      pred_sents = [TokensToSent(CFG.gloss_vocab, s) for s in preds]
      WERS.append(word_error_rate(pred_sents, ref_sents).item())
    
    except IndexError:
      #pdb.set_trace()
      print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")
		
    end = time.time()
    if i % (CFG.print_freq) == 0:
      print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nAvg WER: {np.mean(WERS):.4f}\nTime: {(end - start)/60:.4f} min")
      print("PREDICTIONS:\n", preds)
      #print("###Reference:\n", ref_sents)
    
    #if max(1, i) % 10 == 0:
      #print("###Prediction:\n", pred_sents)
      #print("###Reference:\n", ref_sents)
    #  break

  print(f"FINAL AVG WER: {np.mean(WERS)}")
  return losses, WERS

def validate(model, dataloader, criterion, decoder, CFG):
  losses = []
  model.eval()
  start = time.time()
  WERS = []
  print("################## Starting validation ##################")
  for i, (ipt, trg, trg_len) in enumerate(dataloader):
    with torch.no_grad():
      #print("ipt", ipt.size())
      ipt = ipt.cuda()
      trg = trg.cuda()
      
      out = model(ipt) # add small constant to avoid nan
      #out = out.view(out.size(2), out.size(0), out.size(1)) #### For convolution w. kernel size 1.
      #out = out.view(out.size(1), out.size(0), out.size(2)) #### For nn.Linear layers w. views
      x = out.view(out.size(1), out.size(0), out.size(2))
      ipt_len = torch.full(size=(1,), fill_value=x.size(0), dtype=torch.int32)
      #trg = trg[0]#[:trg_len[0]]
      trg_len = trg_len.to(torch.int32)
      with torch.backends.cudnn.flags(enabled=False):
        loss = criterion(x, 
                        trg, 
                        input_lengths=ipt_len,
                        target_lengths=trg_len)

      losses.append(loss.detach().cpu().item())
      
      #out_d = decoder(torch.exp(out).cpu().view(out.size(0), out.size(2), out.size(1))) # for 1d convolutions
      out_d = decoder(torch.exp(out).cpu())
      
      try:
        preds = [p[0].tokens for p in out_d]
        pred_sents = [TokensToSent(CFG.gloss_vocab, s) for s in preds]
        ref_sents = TokensToSent(CFG.gloss_vocab, trg[0][:trg_len[0]])
        WERS.append(word_error_rate(pred_sents, ref_sents).item())
    
      except IndexError:
        print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")
        #pdb.set_trace()

      end = time.time()
      if i % (CFG.print_freq/2) == 0:
        print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nAvg WER: {np.mean(WERS):.4f}\nTime: {(end - start)/60:.4f} min")
        print("PREDICTIONS:\n", preds)
     
  print(f"FINAL AVG WER: {np.mean(WERS)}")
  return losses, WERS


def save_checkpoint(path, model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS):
  # save a general checkpoint
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'scheduler_state_dict' : scheduler.state_dict(),
              'train_losses' : train_losses,
              'val_losses' : val_losses,
              'train_WERS' : train_WERS,
              'val_WERS' : val_WERS
              }, path)

def TokensToSent(vocab, tokens):
  
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)

if __name__ == '__main__':
  main()