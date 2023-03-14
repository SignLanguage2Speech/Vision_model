import pytorch_lightning as pl
from lightning_ctc import VisualEncoder_lightning

import pandas as pd

def main():
    # init model
    model = VisualEncoder_lightning()
    
    num_gpu = 1
    ckpt_path = None#"some/path/to/my_checkpoint.ckpt"

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=num_gpu,
        # ckpt_path=ckpt_path # full-state checkpoint
        )
    trainer.fit(model)

if __name__ == '__main__':
    main()