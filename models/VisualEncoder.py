from HeadNetwork import HeadNetwork
from S3D_backbone import S3D_backbone
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()