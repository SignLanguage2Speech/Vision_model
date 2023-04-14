import pandas as pd
import os
import numpy as np
from train_datasets.PHOENIXDataset import PhoenixDataset, DataAugmentations, collator, revert_transform_rgb
from torch.utils.data import DataLoader

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

dp = DataPaths()
test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')
PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=1085, split='test')

#data = [PhoenixTest.__getitem__(0), (PhoenixTest.__getitem__(1))]

#ipts, ipt_lens, _, _ = list(zip(*data))

#print(ipt_lens)
#print(ipts[0].size())
#max_len = max(ipt_lens)

#D = [('1', 'hi', 5), ('2', 'hola', 4)]
#z = zip(*D)
#print(list(z))

train_augmentations = DataAugmentations(split_type='train')
dataloaderTest = DataLoader(PhoenixTest, batch_size=2, 
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=lambda data: collator(data, train_augmentations)
                                   )
if __name__ == '__main__':
  import cv2

  for (ipt, ipt_len, trg, trg_len) in dataloaderTest:
    # pdb.set_trace()
    ipt_np = revert_transform_rgb(ipt[1])
    w = h = 224
    c = 3
    fps = 25
    sec = 10
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter('test_TRAIN.mp4', fourcc, float(fps), (w, h))
    for frame_count in range(len(ipt_np)):
      # img_ = np.random.randint(0,255, (h,w,c), dtype = np.uint8)
      img = ipt_np[frame_count].astype(np.uint8)
      # pdb.set_trace()
      video.write(img)
    video.release()

    print("IPTT", ipt.size())
    print(ipt_len)
    print("TRGG", trg.size())
    print(trg_len)
    break