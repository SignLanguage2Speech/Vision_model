import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import S3D, SepConv3d, BasicConv3d, Mixed_3b, Mixed_3c, Mixed_4b, Mixed_4c, Mixed_4d, Mixed_4e, Mixed_4f

class VisualEncoder(nn.Module):
    def __init__(self, n_classes:int):
        super().__init__()

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
        self.head1 = nn.Sequential(nn.Linear(832, 832),
                                  nn.BatchNorm1d(832),
                                  nn.ReLU())
        self.head2 = nn.Sequential(nn.Conv1d(832, 512, kernel_size=3, stride=1, padding=1))

        ### Linear translation layer ###
        self.translation_layer = nn.Sequential(nn.Linear(512, n_classes), 
                                               nn.ReLU(),
                                               nn.Softmax(dim=-1))
        

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (1, y.size(3), y.size(4)), stride=1) # TODO Evaluate if this is equivalent to "spatial pooling".
        y = y.squeeze().T # remove all single dims and transpose
        y = self.head1(y).T
        y = self.head2(y).T
        y = self.translation_layer(y)
        return y


# dim in --> n_frames x 224 x 224 x 3
# dim out --> n_frames/4 x 832

# 12 different base.x 
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