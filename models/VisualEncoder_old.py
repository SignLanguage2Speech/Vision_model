from models.HeadNetwork_old import HeadNetwork
from models.S3D_backbone import S3D_backbone
from models.utils import WeightsLoader
import torch

class VisualEncoder(torch.nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()

        self.CFG = CFG
        self.backbone = S3D_backbone(self.CFG)
        self.head = HeadNetwork(self.CFG)
        
        if CFG.checkpoint_path is not None:
            print("Loading entire state dict directly...")
            checkpoint = torch.load(CFG.checkpoint_path)['model_state_dict']
            self.load_state_dict(checkpoint)
            print("Succesfully loaded")
        
        else:
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
        
        if CFG.freeze_block > 0:
            self.backbone.freeze()
        else:
            print("Everything unfrozen")
        self.set_train()

    def set_train(self):
        block2idx = {0 : 0,
                     1 : 1,
                     2 : 4,
                     3 : 7,
                     4: 13,
                     5: 16}
        if self.CFG.freeze_block < 5:
            for i in range(block2idx[self.CFG.freeze_block], block2idx[self.CFG.use_block]):
                for name, param in self.backbone.base[i].named_parameters():
                    param.requires_grad = True
                self.backbone.base[i].train()
        
        self.head.train()
        for name, param in self.head.named_parameters():
            param.requires_grad = True

    def forward(self, x, vid_lens):
        x, mask = self.backbone(x, vid_lens)
        gloss_probs, gloss_reps = self.head(x, None)
        return gloss_probs, gloss_reps
    