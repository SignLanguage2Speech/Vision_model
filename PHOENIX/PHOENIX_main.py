import os
import pandas as pd
from PHOENIXDataset import PhoenixDataset
from s3d_backbone import VisualEncoder

annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

Backbone = VisualEncoder(2000)

train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
TrainDataset = PhoenixDataset(train, features_path, vocab_size=2000)

img, trg = TrainDataset.__getitem__(10)
