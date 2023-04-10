import torch
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model=512, N=10000) -> None:
        pos = list(range(0, N+1))
        
        self.PE = torch.zeros(1, N, d_model)

        self.PE[:, :, 0::2] = torch.sin(A)
        self.PE[:, :, 1::2] = torch.cos(A)



    
    def forward(self, x):
        return x + self.PE
        
        