import pytorch_lightning as pl
from lightning_s3d import S3D_lightning

import pandas as pd
from main import DataPaths

def main():
    dp = DataPaths()
    df = pd.read_csv(dp.wlasl_labels)
    n_classes = len(set(df['gloss'])) #2000

    # init model
    model = S3D_lightning(n_classes)
    
    trainer = pl.Trainer(accelerator='gpu', devices=2)
    # trainer = pl.Trainer(accelerator='gpu', devices=1)
    trainer.fit(model)

if __name__ == '__main__':
    main()