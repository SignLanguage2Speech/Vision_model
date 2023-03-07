##### Dataset class for Phoenix #####
import os
from Preprocess_PHOENIX import preprocess_df
import torch
import torchvision
from torch.utils import data
import numpy as np
import pandas as pd
import subprocess
from PIL import Image


def transform_rgb(video):
  ''' stack & normalization '''
  video = np.concatenate(video, axis=-1)
  video = torch.from_numpy(video).permute(2, 0, 1).contiguous().float()
  video = video.mul_(2.).sub_(255).div(255)
  out = video.view(-1,3,video.size(1),video.size(2)).permute(1,0,2,3)
  return out

def revert_transform_rgb(clip):
  clip = clip.permute(1, 2, 3, 0)
  clip = clip.contiguous().view(1, -1, clip.size(1), clip.size(2)).squeeze(0)
  clip = clip.mul_(255).add_(255).div(2)
  clip = clip.view(-1, clip.size(1), clip.size(2), 3)
  return clip.numpy()

############# Data augmentations #############
class DataAugmentations:
  def __init__(self):
    self.H_out = 224
    self.W_out = 224

  # flip all images in video horizontally with 50% probability
  def HorizontalFlip(self, imgs):
    p = np.random.randint(0, 2)
    if p < 2: # TODO update this value after testing !!!
      imgs = torchvision.transforms.functional.hflip(imgs)
    return imgs
  
  # random 224 x 224 crop (same crop for all images in video)
  def RandomCrop(self, imgs):
    crop = torchvision.transforms.RandomCrop((self.H_out, self.W_out), padding = 0, padding_mode='constant')
    return crop(imgs)
  
  # 224 x 224 center crop (all images in video)
  def CenterCrop(self, imgs):
    crop = torchvision.transforms.CenterCrop((self.H_out, self.W_out))
    return crop(imgs)
  
  # randomly rotate all images in video with +- 5 degrees.
  def RandomRotation(self, imgs):
    rotate = torchvision.transforms.RandomRotation(5, expand=False, fill=0.0, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
    return rotate(imgs)

def upsample(images, seq_len):
  images_org = images.detach().clone() # create a clone of original input
  if seq_len / images.size(1) >= 2: # check if image needs to be duplicated
    repeats = int(np.floor(seq_len / images.size(1)) - 1) # number of concats
    for _ in range(repeats):
      images = torch.cat((images, images_org), dim=1) # concatenate images temporally
  
  if seq_len > images.size(1):
    start_idx = np.random.randint(0, images_org.size(1) - (seq_len - images.size(1))) 
    stop_idx = start_idx + (seq_len - images.size(1))
    images = torch.cat((images, images_org[:, start_idx:stop_idx]), dim=1)

  return images

def downsample(images, seq_len):
  start_idx = np.random.randint(0, images.size(1) - seq_len)
  stop_idx = start_idx + seq_len
  return images[:, start_idx:stop_idx]


def load_imgs(ipt_dir):
  image_names = os.listdir(ipt_dir)
  N = len(image_names)
  imgs = np.empty((N, 210, 260, 3))

  for i in range(N):
    imgs[i,:,:,:] = np.asarray(Image.open(os.path.join(ipt_dir, image_names[i])))
  
  return imgs
  


class PhoenixDataset(data.Dataset):
    def __init__(self, df, ipt_dir, vocab_size, seq_len=64, train=True):
        super().__init__()
        self.df = preprocess_df(df, save=False, save_name=None)
        self.ipt_dir = ipt_dir
        self.seq_len = seq_len
        self.train = train
        self.vocab_size = vocab_size
        self.video_folders = list(self.df['name'])

    def __getitem__(self, idx):

        ### Assumes that within a sample (id column in df) there is only one folder named '1' ###
        # TODO Check that this holds!
        image_folder = os.path.join(self.ipt_dir, self.video_folders[idx])
        images = load_imgs(image_folder)

        if self.train:
          images = transform_rgb(images) # convert to tensor, reshape and normalize 
          # resize images
          # take a crop in range [0.7, 1.]
          # frame rate augmentation

          # check if we need to upsample
          if self.seq_len > images.size(1): 
            images = upsample(images, self.seq_len)
          # check if we need to downsample
          elif self.seq_len < images.size(1):
            images = downsample(images, self.seq_len)
        
        else:
           images = transform_rgb(images)
           images = self.DataAugmentation.HorizontalFlip(images)
           images = self.DataAugmentations.CenterCrop(images)

        # make a one-hot vector for target class
        sent_len = len(self.df.iloc[idx]['annotation'].split(' '))

        trg = torch.zeros((sent_len, self.vocab_size)) # 2000 unique words
        for i in range(sent_len):
           trg[i]
        gloss_idx = self.df.iloc[idx]['label']
        trg[gloss_idx] = 1
        
        return images, trg

    
    def __len__(self):
        return len(self.df)
    
