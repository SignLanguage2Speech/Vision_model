from models.VisualEncoder import VisualEncoder
from models.S3D_backbone import S3D_backbone
import torch
import time
import numpy as np
import pandas as pd
from datasets.WLASLDataset import WLASLDataset
from torch.utils.data import DataLoader
from configs.VisualEncoderConfig import cfg


CFG = cfg()
model = VisualEncoder(CFG)
ipt = torch.zeros(1, 3, 100, 224, 224)
out = model(ipt)
print(out.size())