##### Dataset class for Phoenix #####
import os
from PHOENIX.preprocess_PHOENIX import preprocess_df
import torch
import torchvision
from torch.utils import data
import numpy as np
import pandas as pd
import subprocess
from PIL import Image
import pdb

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
  def __init__(self, split_type=None):
    self.H_upsample = 298
    self.W_upsample = 240
    self.H_out = 224
    self.W_out = 224
    self.split_type = split_type
    self._center_crop = torchvision.transforms.CenterCrop((self.H_out, self.W_out))
    self._random_crop = torchvision.transforms.RandomCrop((self.H_out, self.W_out), padding = 0, padding_mode='constant')
    self._random_rotate = torchvision.transforms.RandomRotation(5, expand=False, fill=0.0, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
    self._upsample_pixels = torch.nn.Upsample(size=(self.H_upsample, self.W_upsample), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)

  def __call__(self, vid):
    if self.split_type == 'train':
      vid = self.UpsamplePixels(vid)
      vid = transform_rgb(vid) # convert to tensor, reshape and normalize
      vid = self.HorizontalFlip(vid)
      vid = self.RandomCrop(vid)
      vid = self.RandomRotation(vid)
    else:
      # apply validation augmentations
      vid = self.UpsamplePixels(vid)
      vid = transform_rgb(vid)
      vid = self.CenterCrop(vid)
    return vid
  
  def UpsamplePixels(self, imgs: np.ndarray):
    
    imgs = self._upsample_pixels(torch.from_numpy(imgs).double().permute(0, 3, 1, 2).contiguous()) # upsample and place color channel as dim 1
    return imgs.permute(0, 2, 3, 1).numpy() # return and revert dim changes

  # flip all images in video horizontally with 50% probability
  def HorizontalFlip(self, imgs):
    p = np.random.randint(0, 2)
    if p < 2: # TODO update this value after testing !!!
      imgs = torchvision.transforms.functional.hflip(imgs)
    return imgs
  
  # random 224 x 224 crop (same crop for all images in video)
  def RandomCrop(self, imgs):
    return self._random_crop(imgs)
  
  # 224 x 224 center crop (all images in video)
  def CenterCrop(self, imgs):
    return self._center_crop(imgs)
  
  # randomly rotate all images in video with +- 5 degrees.
  def RandomRotation(self, imgs):
    return self._random_rotate(imgs)

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
  imgs = np.empty((N, 260, 210, 3))

  for i in range(N):
    imgs[i,:,:,:] = np.asarray(Image.open(os.path.join(ipt_dir, image_names[i])))
  
  return imgs
  
"""
df : Phoenix dataframe (train, dev or test)
ipt dir : Directory with videos associated to df
vocab size : number of unique glosses/words in df
split : 'train', 'dev' or 'test'
"""

class PhoenixDataset(data.Dataset):
    def __init__(self, df, ipt_dir, vocab_size, split='train'):
        super().__init__()

        self.ipt_dir = ipt_dir
        self.split=split
        self.vocab_size = vocab_size

        if self.split == 'train':
          self.df = df
        else:
          self.df = preprocess_df(df, split, save=False, save_name=None)
  
        self.video_folders = list(self.df['name'])

    def __getitem__(self, idx):
        ### Assumes that within a sample (id column in df) there is only one folder named '1' ###
        image_folder = os.path.join(self.ipt_dir, self.split, self.video_folders[idx])
        # images = load_imgs(image_folder)
        image_names = os.listdir(image_folder)
        N = len(image_names)

        ipt_len = torch.Tensor(N)

        # make a one-hot vector for target class
        trg_labels = torch.tensor(self.df.iloc[idx]['gloss_labels'], dtype=torch.int32)
        trg_length = len(trg_labels)

        return image_names, ipt_len, trg_labels, trg_length

    def __len__(self):
        return len(self.df)

def collator(data, data_augmentation):
  """
  Receives list of data (tuple) and data_aug (class wrapping relevant augmentations)
  data tuple = (image_paths, no. of frames in vid, target_labels, no. of target labels)
  1. select indices from image_paths and perform temporal augmentations
  2. load images corresponding to selected indices
  3. perform spatial augmentations
  """
  image_path_lists, vid_lens, trgs,  trg_lens = list(zip(*data))
  max_ipt_len = max(vid_lens)
  max_trg_len = max(trg_lens)

  batch = torch.zeros((len(image_path_lists), 3, max_ipt_len, 224, 224))
  targets = torch.zeros((len(trgs), max_trg_len))

  vids = []
  for image_paths in image_path_lists:
    imgs = np.empty((len(image_paths), 260, 210, 3))
    for i,ipt in enumerate(image_paths):
      imgs[i,:,:,:] = np.asarray(Image.open(ipt))
    vids.append(imgs)

  for i, vid in enumerate(vids):
    # see DataAugmentation.__call__(self, vid)
    data_augmentation(vid)

    if vid.size(1) > max_ipt_len:
      batch[i] = upsample(vid)
    pad = torch.nn.ConstantPad1d((0, max_trg_len - len(trgs[i])), value=-1)
    targets[i] = pad(trgs[i])
  
  return batch, torch.tensor(vid_lens, dtype=torch.int32), targets, torch.tensor(trg_lens, dtype=torch.int32)



  
"""
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

dataloaderTest = DataLoader(PhoenixTest, batch_size=1, 
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collator)

for (ipt, ipt_len, trg, trg_len) in dataloaderTest:
  print("IPTT", ipt.size())
  print(ipt_len)
  print("TRGG", trg.size())
  print(trg_len)

"""
