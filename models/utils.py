import glob
import torch
import torch.nn as nn

"""
Standard Positional Encoding from Attention Is All You Need
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, N=10000) -> None:
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(N, dtype=torch.float32).reshape(-1, 1)
        A = pos / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        self.PE = torch.zeros(1, N, d_model)
        self.PE[:, :, 0::2] = torch.sin(A)
        self.PE[:, :, 1::2] = torch.cos(A)

    def forward(self, x):
        return x + self.PE[:, :x.size(1), :]
    
"""
class MaskedNorm(nn.Module):
    
        #Original Code from:
        #https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    
    def __init__(self, num_features=512, norm_type='sync_batch', num_groups=1):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            #raise ValueError("Please use sync_batch")
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == 'sync_batch':
            self.norm = nn.SyncBatchNorm(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])
"""


class WeightsLoader:
    def __init__(self, sd, weight_filename) -> None:
        self.sd = sd
        self.weight_filename=weight_filename
        
    
    def load(self, verbose=True):
        file = f'weights/{self.weight_filename}'
        weight_type = self.weight_filename.split('/')[0]
        print("Loading weights from: ", file)
        weights = torch.load(file, map_location='cpu')['state_dict']
        for name, param in weights.items():
            #print("name: ", name)
            # fix naming issues in kinetics state dict
            if weight_type == 'Kinetics':
                name = name.strip('module.') # clean for kinetics
                name = name.replace('track', 'tracked') # clean for kinetics
            elif weight_type == 'WLASL':
                name = name.replace('backbone.', '') # clean for WLASL
                name = name.replace('final_fc.1', 'final_fc.0') # clean for WLASL

            if name in self.sd:
                if param.size() == self.sd[name].size():
                    self.sd[name].copy_(param)
                else:
                    if verbose:
                        print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {self.sd[name].size()}")
            else:
                if verbose:
                    print(f"Param {name} not in state dict")