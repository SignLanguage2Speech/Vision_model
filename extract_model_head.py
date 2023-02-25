import numpy as np
import os
import torch
import torch.nn as nn

from utils.load_weigths import load_model_weights
from model import S3D, SepConv3d, BasicConv3d, Mixed_3b, Mixed_3c, Mixed_4b, Mixed_4c, Mixed_4d, Mixed_4e, Mixed_4f

class PrunedS3D(S3D):
    def __init__(self, n_classes:int):
        super(S3D, self).__init__()
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
    def forward(self, x):
        y = self.base(x)
        print(f"dim after self.base: {y.size()}")


# dim in --> n_frames x 224 x 224 x 3
# dim out --> n_frames/4 x 843

# 12 different base.x 

weights_filename = 'S3D_kinetics400.pt'

model = S3D(2000)
model = load_model_weights(model, weights_filename) 


"""
Comment out everything after Mixed_4f in model.py S3D __init__  --> self.base before running the below line.
This will give out 
"""
#with open(r'data/trimmed_s3d_params.txt', 'w') as f:
#    for name in model.state_dict():
#        f.write("%s\n" % name)
