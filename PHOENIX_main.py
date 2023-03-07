import os
import pandas as pd
from PHOENIX.PHOENIXDataset import PhoenixDataset
from PHOENIX.s3d_backbone import VisualEncoder

annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

VisionModel = VisualEncoder(2000)

train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
TrainDataset = PhoenixDataset(train, features_path, vocab_size=2000)

img, trg = TrainDataset.__getitem__(10)
img = img.unsqueeze(0) # no batch size because dataloader is not currently used
print(f"Input size: {img.size()}")

out = VisionModel(img)
print(f"Output size: {out.size()}")