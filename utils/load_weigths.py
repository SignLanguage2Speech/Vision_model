import os, glob
import torch

class WeightsLoader:
    def __init__(self, sd, weight_filename) -> None:
        self.sd = sd
        self.weight_filename=weight_filename
        
    
    def load(self, verbose=True):
        file = f'weights/{self.weight_filename}'
        print("LOADING file: ", file)
        weights = torch.load(file, map_location='cpu')
        for name, param in weights.items():
            # fix naming issues in kinetics state dict
            name = name.strip('module.')
            name = name.replace('track', 'tracked')
            name = name.strip('backbone.')
            if name in self.sd:
                if param.size() == self.sd[name].size():
                    self.sd[name].copy_(param)
                else:
                    if verbose:
                        print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
            else:
                if verbose:
                    print(f"Param {name} not in state dict")



def load_kinetics_weights(model, weight_filename = 'S3D_kinetics400.pt', verbose=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    weights = torch.load(os.path.join('weights', weight_filename), map_location=torch.device(device))
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
                if verbose:
                    print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            if verbose:
                print(f"Param {name} not in state dict")
            
    return model

def load_WLASL_weights(model, weight_filename, verbose=False):
    sd = model.state_dict()
    if '.tar' in weight_filename:
        file = glob.glob('./weights/WLASL/*.pth.tar')
        checkpoint = torch.load(file[0], map_location='cpu')
        weights = checkpoint['state_dict']
    else:
        print("Not implemented yet")
    
    for name, param in weights.items():
        name = name.replace('backbone.', '')
        if name in sd:
            if param.size() == sd[name].size():
                sd[name].copy_(param)
            else:
                if verbose:
                    print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            if verbose:
                print(f"Param {name} not in state dict")

        
def load_PHOENIX_weights(model, path='/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc', verbose=False):
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    #weights = torch.load(), map_location=torch.device(device))
    checkpoint = torch.load(path)
    weights = checkpoint['model_state_dict']
    sd = model.state_dict()
    
    for name, param in weights.items():
        
        if name in sd:
            if param.size() == sd[name].size():
                sd[name].copy_(param)
            else:
                if verbose:
                    print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            if verbose:
                print(f"Param {name} not in state dict")
            
