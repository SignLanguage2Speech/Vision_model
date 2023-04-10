import torch
from models.S3D.model import S3D


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
        


    def load_weights(self, model, ckpt='WLASL'):
        ckpts = ['phoenix', 'wlasl', 'kinetics', 'how2sign']
        assert(ckpt.lower() in ckpts, print(f"{ckpt} is not a valid checkpoint!\n Valid ones are:\n{ckpts}"))
        print(f"Loading weights for {ckpt}")

    def forward(self, x):
        x = self.base(x)
        x = torch.mean(x, dim=[3, 4])
        x = x.transpose(1, 2)
        return x
