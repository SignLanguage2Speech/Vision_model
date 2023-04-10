from models.HeadNetwork import HeadNetwork
from models.S3D_backbone import S3D_backbone
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self, n_classes, CFG) -> None:
        super().__init__()

        self.backbone = S3D_backbone(CFG)
        self.head = HeadNetwork(n_classes=400, input_size=832, hidden_size=512, 
                                ff_size=2048, ff_kernel_size=3, residual_connection=False)
    
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

    def compute_loss(self, log_probs, ipt_lens, trg, trg_lens):
        # N, T, K --> T, N, K 
        loss = self.ctc_loss(log_probs.permute(1,0,2),
                             trg,
                             ipt_lens,
                             trg_lens)
        return loss / log_probs.size(0) # divide with batch size
    
    def decode(self, logits, beam_size, ipt_lens):

        pass
    

    def forward(self, x):
        x = self.backbone(x)
        logits = self.head(x)
        