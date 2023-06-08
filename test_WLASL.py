import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from WLASL_main import validate, load_checkpoint
from train_datasets.WLASLDataset import WLASLDataset
from models.S3D.model import S3D

class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL_labels.csv"

class cfg:
  def __init__(self):
    self.load_path = '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc' # ! Fill empty string with model file name
    self.use_block = 5
    # self.checkpoint = True # start from checkpoint set in self.load_path
    self.print_freq = 100
    self.multipleGPUs = False
    # for data augmentation
    self.crop_size = 224
    self.seq_len = 64
    self.num_workers = 8
    self.top_k = 10

def main():
    ############## initialization ##############
    dp = DataPaths() # DATA PATHS
    CFG = cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############## load data ##############
    df = pd.read_csv(dp.wlasl_labels)
    img_folder = dp.wlasl_videos
    WLASLtrain = WLASLDataset(df.loc[df['split']=='train'], img_folder, seq_len=CFG.seq_len,train=True, grayscale=False)
    WLASLval = WLASLDataset(df.loc[df['split']=='val'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)
    WLASLtest = WLASLDataset(df.loc[df['split']=='test'], img_folder, seq_len=CFG.seq_len, train=False, grayscale=False)

    
    dataloaderVal = DataLoader(WLASLval, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    # TODO actually use this 🤡
    dataloaderTest = DataLoader(WLASLtest, batch_size=1, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers)
    ############## load checkpoint ##############

    model = S3D(CFG.use_block).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) ### not used
    
    model, optimizer, latest_epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(CFG.load_path, model, optimizer)
    CFG.start_epoch = latest_epoch

    criterion = torch.nn.CrossEntropyLoss().cuda()
    ############## run validation ##############
    validate(model, dataloader=dataloaderTest, criterion=criterion, CFG=CFG)

if __name__ == '__main__':
  main()


