import numpy as np
import time
import torch
import torch.nn as nn
from s3d_backbone import VisualEncoder

class cfg:
    def __init__(self):
        self.start_epoch = 0
        self.n_epochs = 80
        self.init_lr = -1000 # TODO look up value
        self.momentum = -1000 # TODO look up value
        self.epsilon = -1000 


def train(model, optimizer, criterion):
    model.train()
    start = time.time()
