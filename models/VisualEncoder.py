from models.HeadNetwork import HeadNetwork
from models.S3D_backbone import S3D_backbone
from models.utils import WeightsLoader
import torch

class VisualEncoder(torch.nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()

        self.backbone = S3D_backbone(CFG)
        self.head = HeadNetwork(CFG)
        #self.weights_loader = WeightsLoader(sd = self.state_dict(), weight_filename=CFG.weights_filename)
        
        try:
            print("Loading entire state dict directly...")
            checkpoint = torch.load(CFG.weights_filename)['model_state_dict']
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
        
        #self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
    """
    def compute_loss(self, log_probs, ipt_lens, trg, trg_lens):
        # N, T, K --> T, N, K 
        loss = self.ctc_loss(log_probs.permute(1,0,2),
                             trg,
                             ipt_lens,
                             trg_lens)
        return loss / log_probs.size(0) # divide with batch size

    def decode(self, logits, beam_size, ipt_lens):

        pass
    
    """
    
    def forward(self, x):
        x = self.backbone(x)
        gloss_probs = self.head(x)
        return gloss_probs