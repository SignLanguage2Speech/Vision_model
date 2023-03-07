import pytorch_lightning as pl
from lightning_ctc import VisualEncoder_lightning

import pandas as pd

def main():
    # init model
    model = VisualEncoder_lightning()
    
    # trainer = pl.Trainer(accelerator='gpu', devices=2)
    trainer = pl.Trainer()
    trainer.fit(model)

if __name__ == '__main__':
    main()