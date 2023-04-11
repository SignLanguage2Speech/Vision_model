from models.HeadNetwork import HeadNetwork
from models.S3D_backbone import S3D_backbone
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()

        self.backbone = S3D_backbone(CFG)
        self.head = HeadNetwork(n_classes=CFG.n_classes, input_size=CFG.input_size, hidden_size=CFG.hidden_size, 
                                ff_size=CFG.ff_size, ff_kernel_size=CFG.ff_kernel_size, residual_connection=CFG.residual_connection)
    
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

    def compute_loss(self, log_probs, ipt_lens, trg, trg_lens):
        # N, T, K --> T, N, K 
        loss = self.ctc_loss(log_probs.permute(1,0,2),
                             trg,
                             ipt_lens,
                             trg_lens)
        return loss / log_probs.size(0) # divide with batch size
    
    def load_weights(self):
        print(f"Loading weights from {self.CFG.weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=True)

    def decode(self, logits, beam_size, ipt_lens):

        pass
    

    def forward(self, x):
        x = self.backbone(x)
        logits = self.head(x)
        