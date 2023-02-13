import os
import numpy as np
import torch 
from PIL import Image
import torch.utils.data as data

def transform_rgb(snippet):
  #print(f"Number of images being transformed: {len(snippet)}")
  #print(f"Original image size: {snippet[0].shape}")
  ''' stack & normalization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  #print(f"Post transformation size: {out.size()}")
  return out

def transform_gray(snippet):
  #print(f"Number of images being transformed: {len(snippet)}")
  #print(f"Original image size: {snippet[0].shape}")
  ''' stack & normalization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,1,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  #print(f"Post transformation size: {out.size()}")
  return out

def revert_transform_rgb(clip):
  clip = clip.squeeze(0).permute(1, 2, 3, 0)
  clip = clip.contiguous().view(1, -1, clip.size(1), clip.size(2)).squeeze(0)
  clip = clip.mul_(255).add_(255).div(2)
  clip = clip.view(-1, clip.size(1), clip.size(2), 3)
  return clip.numpy()


class WLASLDataset(data.Dataset):
  def __init__(self, df, img_folder, seq_len=None, grayscale=False):
    super().__init__()
    self.df = df
    self.img_folder = img_folder
    self.video_names = os.listdir(self.img_folder)
    self.grayscale = grayscale
    self.seq_len = seq_len
  def __getitem__(self, idx):
    video_path = os.path.join(self.img_folder, self.video_names[idx])
    
    if self.grayscale:
      images = transform_gray([np.expand_dims(np.asarray(Image.open(os.path.join(video_path, f)).convert('L')),2) for f in os.listdir(video_path)])
    else:
      images = transform_rgb([np.asarray(Image.open(os.path.join(video_path, f))) for f in os.listdir(video_path)])
    
    if self.seq_len is not None:

      if self.seq_len < images.size(2): #upsample to reach seq_len
        print("not implemented yet...")
      elif self.seq_len > images.size(2): #downsample to reach seq_len
        start_idx = np.random.randint(0, images.size(2)-self.seq_len)
        images = images[:][:][start_idx:]

    trg = self.df['gloss'][idx]
    #trg = self.df['label'][idx]
    return images, trg
  
  def __len__(self):
    return len(self.video_names)