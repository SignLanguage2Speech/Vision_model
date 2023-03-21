import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import S3D, SepConv3d, BasicConv3d, Mixed_3b, Mixed_3c, Mixed_4b, Mixed_4c, Mixed_4d, Mixed_4e, Mixed_4f

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