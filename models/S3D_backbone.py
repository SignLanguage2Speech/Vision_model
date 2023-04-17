import torch
import torch.nn.functional as F
from models.S3D.model import S3D
from models.utils import WeightsLoader
import pdb

class S3D_backbone(S3D):
    def __init__(self, CFG) -> None:
        super(S3D_backbone, self).__init__(use_block=CFG.use_block)

        self.CFG = CFG
        self.frozen_modules = []
        self.freeze_block = CFG.freeze_block
        self.use_block = CFG.use_block

        if CFG.checkpoint_path == None:
            self.weightsLoader = WeightsLoader(self.state_dict(), CFG.backbone_weights_filename)
        #    self.load_weights(CFG.verbose)

    
    def load_weights(self, verbose):
        print(f"Loading weights from {self.CFG.backbone_weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=verbose)

    def freeze(self):
        block2idx = {1 : 1,
                     2 : 4,
                     3 : 7,
                     4: 13,
                     5: 16}
        # freeze blocks 1... 5
        print(f"Freezing up to block {self.freeze_block} in S3D backbone")
        for i in range(block2idx[self.freeze_block]):
            #for m in self.base[i]:
            for name, param in self.base[i].named_parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.base(x)
        if self.CFG.use_block == 5:
            x = F.avg_pool3d(x, (2, x.size(3), x.size(4)), stride=1)
            x = self.final_fc(x.view(-1, x.size(2), x.size(1)))
            return torch.mean(x, 1)

        x = torch.mean(x, dim=[3, 4])
        x = x.transpose(1, 2)
        return x
