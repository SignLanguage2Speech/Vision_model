import os
import torch

# from lightning_ctc import VisualEncoder_lightning
# from visual_encoder import VisualEncoder

import pdb

model_path = '/work3/s204138/bach-models/trained_models/S3D_WLASL-69_epochs-3.397701_loss_0.290858_acc'
    
# sd = torch.load(model_path)

def load_model_weights(model, weight_filename=None, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    # pdb.set_trace()
    weights = torch.load(weight_filename)#, map_location=torch.device(device))
    # pdb.set_trace()
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
    
    return model


# print("creating model")
# model = VisualEncoder(1085)
# print("done creating model")

# model = load_model_weights(model, weight_filename=model_path, verbose=True)

# pdb.set_trace()
# print('')