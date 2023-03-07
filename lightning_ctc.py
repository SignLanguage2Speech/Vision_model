import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from visual_encoder import VisualEncoder

class VisualEncoder_lightning(pl.LightningModule):
    def __init__(self, num_class, load_model=False):
        super().__init__()
        self.cfg = cfg()
        self.dp = DataPaths()
        self.model = VisualEncoder(num_class)
        self.criterion = nn.CTCLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_num):
        x,y = batch
        n_batch = len(x)
        out = self.model(x)
        # pdb.set_trace()
        loss = self.criterion(out, y)

        return {'loss': loss}
    
    def configure_optimizers(self):
        return self.optimizer if self.optimizer is not None else \
            optim.SGD(self.model.parameters(),
                    self.cfg.lr,
                    weight_decay=self.cfg.weight_decay,
                    momentum=self.cfg.momentum)

    def get_dataloader(self, split_type) -> DataLoader:
        raise NotImplementedError("get_dataloader not implemented yet!")
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')