import os
import torch
from model import S3D


weight_filename = 'S3D_kinetics400.pt'

def load_model_weights(n_classes):
    model = S3D(n_classes)
    weights = torch.load(weight_filename)
    sd = model.state_dict()

    for name, param in weights.items():
        # fix naming issues in kinetics state dict
        if "module" in name:
            name = name.strip('module.')
            name = name.replace('track', 'tracked')
        if name in sd:
            if param.size() == sd[name].size():
                sd[name].copy_(param)
            else:
                print(f"name: {name}\npre-trained size {param.size()}\nInit size: {sd[name].size()}")
        else:
            print(f"Param {name} not in state dict")
            
    return model

m = load_model_weights(100)