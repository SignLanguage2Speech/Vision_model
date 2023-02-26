import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.WLASLDataset import transform_rgb 
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
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1) # TODO Evaluate if this is equivalent to "spatial pooling".
        #print(f"dim after self.base: {y.size()}")
        
        return y


# dim in --> n_frames x 224 x 224 x 3
# dim out --> n_frames/4 x 843

# 12 different base.x 

weights_filename = 'S3D_kinetics400.pt'
default_wd = os.getcwd()

model = S3D(2000)
model = load_model_weights(model, weights_filename) 

img_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_images/00333')
os.chdir(img_folder)
ipt = [np.asarray(Image.open(f)) for f in os.listdir(img_folder)]
os.chdir(default_wd)
ipt = transform_rgb(ipt).unsqueeze(0)

model.eval()
out_default = model(ipt)
print(f"default out shape: {out_default.size()}")
model2 = PrunedS3D(2000)
model2 = load_model_weights(model2, weights_filename)
model2.eval()
out = model2(ipt)
print(f'pruned model output: {out.size()}')
"""
Comment out everything after Mixed_4f in model.py S3D __init__  --> self.base before running the below line.
This will give out 
"""
#with open(r'data/trimmed_s3d_params.txt', 'w') as f:
#    for name in model.state_dict():
#        f.write("%s\n" % name)
