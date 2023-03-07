import torch
import pytorch_lightning as pl

import pdb

from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np

from model import S3D
from main import cfg,DataPaths,load_checkpoint
from utils.WLASLDataset import WLASLDataset
from utils.load_weigths import load_model_weights

class S3D_lightning(pl.LightningModule):
    def __init__(self, num_class, load_model=False):
        super().__init__()
        self.cfg = cfg()
        self.dp = DataPaths()
        # self.
        self.model = S3D(num_class)
        if load_model:
            model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(self.cfg.load_path)
            self.model = model
            self.model.optimizer = optimizer
        else:
            self.model = load_model_weights(self.model, 'S3D_kinetics400.pt')
            self.optimizer = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_num):
        x,y = batch
        n_batch = len(x)
        out = self.model(x)
        # pdb.set_trace()
        loss = self.criterion(out, y)

        # _, preds = out.topk(1, 1, True, True) # top 1
        _, preds = out.topk(5, 1, True, True) # top 5 
        acc = 0
        for i in range(len(preds)):
            for j in range(len(preds[1])):
                if preds[i][j] == np.where(y.cpu()[i] == 1)[0][0]:
                    acc += 1
                    break

        print('Correct preds in batch:', acc)

        return {'loss': loss}
    
    def configure_optimizers(self):
        return self.optimizer if self.optimizer is not None else \
            optim.SGD(self.model.parameters(),
                    self.cfg.lr,
                    weight_decay=self.cfg.weight_decay,
                    momentum=self.cfg.momentum)

    def get_dataloader(self, split_type) -> DataLoader:
        df = pd.read_csv(self.dp.wlasl_labels)
        img_folder = self.dp.wlasl_videos

        is_train = split_type == 'train'
        WLASL_split = WLASLDataset(df.loc[df['split']==split_type], img_folder, seq_len=self.cfg.seq_len, train=is_train, grayscale=False)
        
        return DataLoader(WLASL_split, batch_size=self.cfg.batch_size, 
                                        shuffle=True,
                                        num_workers=self.cfg.num_workers)
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')


