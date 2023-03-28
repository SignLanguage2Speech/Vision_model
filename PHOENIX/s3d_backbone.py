import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import S3D, SepConv3d, BasicConv3d, Mixed_3b, Mixed_3c, Mixed_4b, Mixed_4c, Mixed_4d, Mixed_4e, Mixed_4f

### VisualEncoder with convolutions kernel size 1
"""
class VisualEncoder(nn.Module):
    def __init__(self, n_classes:int):
        super(VisualEncoder, self).__init__()

        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f())
        
        ### Model head ###
        self.head1 = nn.Sequential(nn.Conv1d(832, 832, kernel_size=1, padding=0), # temporal linear layer
                                   nn.BatchNorm1d(num_features=832),
                                   nn.ReLU())
        self.head2 = nn.Sequential(nn.Conv1d(832, 512, kernel_size=3, stride=1, padding=1))

        ### Linear translation layer ###
        self.translation_layer = nn.Sequential(nn.Conv1d(512, n_classes, kernel_size=1, padding=0), # projection block
                                               nn.ReLU(),
                                               nn.Softmax(dim=1))
        

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (1, y.size(3), y.size(4)), stride=1) # TODO Evaluate if this is equivalent to "spatial pooling".
        out = y.view(-1, y.size(1), y.size(2)) # collapse singleton dimensions
        out = self.head1(out)
        out = self.head2(out)
        out = self.translation_layer(out)
        return out
"""




### Visual encoder with nn.Linear and views
class VisualEncoder(nn.Module):
    def __init__(self, n_classes:int):
        super(VisualEncoder, self).__init__()

        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f())
        
        ### Model head ###
        self.head1_1 = nn.Linear(832, 832)# temporal linear layer
        self.head1_2 = nn.Sequential(nn.BatchNorm1d(num_features=832),
                                     nn.ReLU())
        
        self.head2 = nn.Sequential(nn.Conv1d(832, 512, kernel_size=3, stride=1, padding=1))

        ### Linear translation layer ###
        self.translation_layer = nn.Sequential(nn.Linear(512, n_classes), # projection block
                                               nn.ReLU(),
                                               nn.LogSoftmax(dim=-1))
        

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (1, y.size(3), y.size(4)), stride=1) # TODO Evaluate if this is equivalent to "spatial pooling".
        y2 = y.view(-1, y.size(2), y.size(1)) # collapse singleton dimensions
        y2 = self.head1_1(y2)
        y3 = self.head1_2(y2.view(-1, y2.size(2), y2.size(1)))
        y3 = self.head2(y3)
        out = self.translation_layer(y3.view(-1, y3.size(2), y3.size(1)))
        return out
    
"""
import pandas as pd
from utils.load_weigths import load_model_weights
from utils.WLASLDataset import WLASLDataset
import torch.utils.data as data

class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL_labels.csv"

dp = DataPaths()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisualEncoder(2001)
model = load_model_weights(model, 'S3D_kinetics400.pt')#.to(device)
print("model loaded")

df = pd.read_csv(dp.wlasl_labels)
img_folder = dp.wlasl_videos

# get datasets
WLASLtrain = WLASLDataset(df.loc[df['split']=='train'], img_folder, seq_len=128, train=True, grayscale=False)

dataloaderTrain = data.DataLoader(WLASLtrain, batch_size=1, 
                                   shuffle=False,
                                   num_workers=0)

with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloaderTrain):
        #ipt = ipt.cuda()
        #trg = trg.cuda()
        out = model(ipt)
        print(f"Out size: {out.size()}")
        if i == 1:
           break
"""


   
"""
weights_filename = 'S3D_kinetics400.pt'
default_wd = os.getcwd()
model = VisualEncoder(2001)
model = load_model_weights(model, weights_filename)

img_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_images/00333')
os.chdir(img_folder)
ipt = [np.asarray(Image.open(f)) for f in os.listdir(img_folder)]
os.chdir(default_wd)
ipt = transform_rgb(ipt).unsqueeze(0)
print(f"IPT size: {ipt.size()}")

model.eval()
out = model(ipt).detach().numpy()
print(f'pruned model output: {out.shape}')


import pandas as pd
from CTC_decoder.beam_search import beam_search

df = pd.read_csv('data/WLASL/WLASL_labels.csv')
vocab = list(set(df['gloss'])) 

e = beam_search(out, vocab)
print(e)
"""
