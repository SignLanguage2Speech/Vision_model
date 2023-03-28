import pytorch_lightning as pl
from lightning_ctc import VisualEncoder_lightning

import pickle

import pandas as pd

def main():
    # ckpt_path = "/work3/s204503/bach-models/ctc-models/lightning_logs/version_12/checkpoints/epoch=67-step=80444.ckpt" # full-state checkpoint
    # ckpt_path = "/work3/s204503/bach-models/ctc-models/lightning_logs/version_15/checkpoints/epoch=67-step=80444.ckpt" # full-state checkpoint
    ckpt_path = None

    # init model
    model = VisualEncoder_lightning() if ckpt_path is None else VisualEncoder_lightning.load_from_checkpoint(ckpt_path)

    # pickle.dump(model, './PICKLE_CTCDEC_TEST')
    
    num_gpu = -1

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=num_gpu,
        default_root_dir="/work3/s204503/bach-models/ctc-models",
        # strategy='ddp', replace_sampler_ddp=False
        )
    trainer.fit(model)

if __name__ == '__main__':
    main()