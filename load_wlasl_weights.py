import os
import torch

import pdb

def load_model_weights(model, weight_filename=None, verbose=False):
    weights = torch.load(weight_filename)
    sd = model.state_dict()

    for name, param in weights['model_state_dict'].items():
        if name in sd:
            if param.size() == sd[name].size():
                sd[name].copy_(param)
            else:
                if verbose:
                    print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            if verbose:
                print(f"Param {name} not in state dict")