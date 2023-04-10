import torch
import torch.nn.functional as F
from models.S3D.model import S3D
from models.utils import WeightsLoader

class S3D_backbone(S3D):
    def __init__(self, CFG) -> None:
        super(S3D_backbone, self).__init__(use_block=CFG.use_block)
        
        self.frozen_modules = []
        self.freeze_block = CFG.freeze_block
        self.use_block = CFG.use_block
        self.CFG = CFG
        self.weightsLoader = WeightsLoader(self.state_dict(), CFG.weights_filename)
        
        print("LOADING WEIGHTS")
        print(CFG.weights_filename)
        self.load_weights()
        # freeze blocks 1... 5
        if self.freeze_block > 0:
            for i in range(len(self.base)): # 0, 1, ... 16
                print("Not implemented...")

    def load_weights(self):
        print(f"Loading weights from {self.CFG.weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=True)

    def forward(self, x):
        x = self.base(x)
        if self.CFG.use_block == 5:
            x = F.avg_pool3d(x, (2, x.size(3), x.size(4)), stride=1)
            x = self.final_fc(x.view(-1, x.size(2), x.size(1)))
            return torch.mean(x, 1)

        x = torch.mean(x, dim=[3, 4])
        x = x.transpose(1, 2)
        return x
