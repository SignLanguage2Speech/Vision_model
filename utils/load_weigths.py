import os
import torch

def load_model_weights(model, weight_filename = 'S3D_kinetics400.pt'):
    
    weights = torch.load(os.path.join('weights', weight_filename))
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
                print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            print(f"Param {name} not in state dict")
            
    return model
