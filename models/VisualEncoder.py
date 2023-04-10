from HeadNetwork import HeadNetwork
from S3D_backbone import S3D_backbone
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = S3D_backbone(use_block=4, freeze_block=1)
        self.head = HeadNetwork(n_classes=400, input_size=832, hidden_size=512, 
                                ff_size=2048, ff_kernel_size=3, residual_connection=True)
    
    def forward(self, x):
        x = self.backbone(x)
        logits = self.head(x)