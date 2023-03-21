import pytorch_lightning as pl
from lightning_ctc import VisualEncoder_lightning

import pickle

import pandas as pd

def main():
    ckpt_path = "/work3/s204503/bach-models/ctc-models/lightning_logs/version_9/checkpoints/epoch=2-step=1185.ckpt" # full-state checkpoint

    # init model
    model = VisualEncoder_lightning() if ckpt_path is None else VisualEncoder_lightning.load_from_checkpoint(ckpt_path)

    # pickle.dump(model, './PICKLE_CTCDEC_TEST')
    
    num_gpu = -1

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=num_gpu,
        default_root_dir="/work3/s204503/bach-models/ctc-models"
        )
    trainer.fit(model)

if __name__ == '__main__':
    main()