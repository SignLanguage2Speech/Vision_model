from models.VisualEncoder import VisualEncoder
from models.S3D_backbone import S3D_backbone
import torch
import time
import numpy as np
import pandas as pd
from datasets.WLASLDataset import WLASLDataset
from torch.utils.data import DataLoader

class cfg:
    def __init__(self) -> None:
        # S3D backbone
        self.use_block = 5 # use everything except lass block
        self.freeze_block = 0 # 0 is unfrozen
        self.weights_filename='WLASL/epoch299.pth.tar'
        self.seq_len=64


class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL_labels.csv"


def main():
  dp = DataPaths() # DATA PATHS
  CFG = cfg()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ############## load data ##############
  df = pd.read_csv(dp.wlasl_labels)
  img_folder = dp.wlasl_videos

  # get datasets
  WLASLval = WLASLDataset(df.loc[df['split']=='val'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
  WLASLtest = WLASLDataset(df.loc[df['split']=='test'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)

  ############## initialize model and optimizer ##############
  n_classes = len(set(df['gloss'])) #2000
  model = S3D_backbone(CFG).to(device)
  
  dataloaderVal = DataLoader(WLASLval, batch_size=1, 
                                    shuffle=True,
                                    num_workers=CFG.num_workers)
  # TODO actually use this 🤡
  dataloaderTest = DataLoader(WLASLtest, batch_size=1, 
                                    shuffle=True,
                                    num_workers=CFG.num_workers)
  print(f"Model is on device: {device}")                 
  
  criterion = torch.nn.CrossEntropyLoss().cuda()

  # run validation loop
  val_loss, val_acc = validate(model, dataloaderVal, criterion, CFG)
  print(f"Evaluation finished!\nLoss: {val_loss:.4f}\nAcc: {val_acc:.4f}")


def validate(model, dataloader, criterion, CFG):
  losses = []
  model.eval()
  start = time.time()
  acc = 0
  print("################## Starting validation ##################")
  for i, (ipt, trg) in enumerate(dataloader):
    with torch.no_grad():

      ipt = ipt.cuda()
      trg = trg.cuda()
      #print(f"Input size: {ipt.size()}")

      out = model(ipt)
      loss = criterion(out, trg)
      losses.append(loss.cpu())

      _, preds = out.topk(1, 1, True, True)
      for j in range(len(preds)):
        if preds[j] == np.where(trg.cpu()[j] == 1)[0][0]:
          acc += 1
      
      end = time.time()
      if i % (CFG.print_freq/2) == 0:
        print(f"Iter: {i}/{len(dataloader)}\nAvg loss: {np.mean(losses):.6f}\nCurrent accuracy: {acc /(i+1):.4f}\nTime: {(end - start)/60:.4f} min")

  acc = acc/len(dataloader.dataset)
  return losses, acc