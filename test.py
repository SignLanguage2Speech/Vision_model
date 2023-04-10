from models.VisualEncoder import VisualEncoder
import torch

class cfg:
    def __init__(self) -> None:

        # S3D backbone
        self.use_block = 4 # use everything except lass block
        self.freeze_block = 0 # 0 is unfrozen
CFG = cfg()

x = torch.ones(1, 3, 64, 224, 224)
model = VisualEncoder(400, CFG)

out = model.backbone(x)
print("OUT", out.size())
mask = torch.zeros(16)
out2 = model.head(out, mask)
print(out2.size())