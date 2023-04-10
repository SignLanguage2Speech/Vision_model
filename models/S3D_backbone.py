import torch.nn as nn
from S3D.model import S3D



class S3D_backbone(S3D):
    def __init__(self, use_block=4, freeze_block=0) -> None:
        super(S3D_backbone, self).__init__(num_class=400, use_block=use_block)
        
        self.frozen_modules = []
        self.freeze_block = freeze_block
        self.use_block = use_block

        # freeze blocks 1... 5
        if freeze_block > 0:
            for i in range(len(self.base)): # 0, 1, ... 16
                print("Not implemented...")
        




    def forward(self, x):
        out = self.base(x)
        return out
