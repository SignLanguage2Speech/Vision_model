from models.HeadNetwork import HeadNetwork
from models.S3D_backbone import S3D_backbone
from models.utils import WeightsLoader
import torch

class VisualEncoder(torch.nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()

        self.backbone = S3D_backbone(CFG)
        self.head = HeadNetwork(CFG)
        
        try:
            print("Loading entire state dict directly...")
            checkpoint = torch.load(CFG.checkpoint_path)['model_state_dict']
            self.load_state_dict(checkpoint)
            print("Succesfully loaded")
        
        except:
            print("Loading state dicts individually")

            if CFG.backbone_weights_filename != None:
                print("Loading weights for S3D backbone")
                self.backbone.weightsLoader.load(CFG.verbose)
            else:
                print("Training backbone from scratch")
            if CFG.head_weights_filename != None:
                print("Loading weights for head network")
                self.head.weightsLoader.load(CFG.verbose)
            else:
                print("Training head network from scratch")
    
    def forward(self, x):
        x = self.backbone(x)
        gloss_probs = self.head(x)
        return gloss_probs