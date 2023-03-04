import numpy as np
import pandas as pd
import os

import torch
import torchvision

from model import S3D
from utils.WLASLDataset import WLASLDataset, video2array, transform_rgb
from torch.utils.data import DataLoader
from PIL import Image

class DataPaths:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL2000"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL_labels.csv"

class DataPaths_dummy:
  def __init__(self):
    self.wlasl_videos = "/work3/s204138/bach-data/WLASL/WLASL100"
    self.wlasl_labels = "/work3/s204138/bach-data/WLASL/WLASL100_labels.csv"

class DataPaths_local:
  def __init__(self):
    self.wlasl_videos = "/data/WLASL/WLASL_videos"
    self.wlasl_labels = "/data/WLASL/WLASL_labels.csv"

dp = DataPaths()
df = pd.read_csv(dp.wlasl_labels)


"""
Check whether all dataloader correspond to their index in the dataset
"""
def AssertLabelsWork(dp):

  ### Load data ###
  df = pd.read_csv(dp.wlasl_labels)
  ipt_dir = dp.wlasl_videos
  WLASL = WLASLDataset(df, ipt_dir, train=True)
  dataloader = DataLoader(WLASL, batch_size=1, shuffle=False, num_workers=4)

  for idx, (ipt, trg) in enumerate(dataloader):

    trg_idx = np.where(trg.squeeze(0) == 1)[0][0]
    if not trg_idx == df.loc[idx, 'label']:
      print(f"MISMATCH! At index {idx}\nTarget label: {trg_idx}\nDf label: {df.loc[idx, 'label']}")

    if idx % 250 == 0:
      print(f"iter {idx}/{len(dataloader)}")

#AssertLabelsWork(dp)

"""
Check that all image transformations work
"""

# flip with 100% probability
def HorizontalFlip(imgs):
  imgs = torchvision.transforms.functional.hflip(imgs)
  return imgs

def CenterCrop(self, imgs):
    crop = torchvision.transforms.CenterCrop((224, 224))
    return crop(imgs)


class WLASLDeterministicDataset(WLASLDataset):
  def __init__(self, df, input_dir, seq_len=64, train=True, grayscale=False):
    super().__init__(df, input_dir, seq_len, train, grayscale)
  
  def __getitem__(self, idx):

    vname = str(self.video_names[idx])
    vname = "0" * (5-len(vname)) + vname + '.mp4'

    ipt = video2array(vname, self.input_dir) # convert video to np array
    images = transform_rgb(ipt)
    ### Data Augmentations ###

    images = HorizontalFlip(images)
    images = CenterCrop(images)

    trg = torch.zeros(self.vocab_size) # 2000 unique words
    gloss_idx = self.df.iloc[idx]['label']
    trg[gloss_idx] = 1
    
    return images, trg


def CheckImgTransformations(dp):
  
  df = pd.read_csv(dp.wlasl_labels)
  ipt_dir = dp.wlasl_videos
  WLASL = WLASLDeterministicDataset(df, ipt_dir)
  dataloader = DataLoader(WLASL, batch_size=1, shuffle=False, num_workers=4)

  local_img_folder = 'data/WLASL/WLASL_images'
  video_names = os.listdir(local_img_folder)

  for idx, (ipt, trg) in enumerate(dataloader):

    imgs_path = os.path.join(local_img_folder, video_names[idx])
    image_names = os.listdir(imgs_path)
    local_images = [np.asarray(Image.open(os.path.join(imgs_path, e))) for e in image_names]

    # MAKE SURE TRANSFORMS ARE THE SAME AS IN __GETITEM__
    local_images = transform_rgb(local_images)
    local_images = HorizontalFlip(local_images)
    local_images = CenterCrop(local_images)

    if not local_images == ipt:
      print(f"MISMATCH at index {idx}")





    













    



