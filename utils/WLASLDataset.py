import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import subprocess
import shlex
import torchvision

import pdb

"""
Function that converts array of RGB images to [batch_size x 3 x n_frames x H x W]
"""

def transform_rgb(video):
  ''' stack & normalization '''

  video = np.concatenate(video, axis=-1)
  video = torch.from_numpy(video).permute(2, 0, 1).contiguous().float()
  video = video.mul_(2.).sub_(255).div(255)
  out = video.view(-1,3,video.size(1),video.size(2)).permute(1,0,2,3)
  return out


def transform_gray(snippet):
  print(f"Number of images being transformed: {len(snippet)}")
  print(f"Original image size: {snippet[0].shape}")
  ''' stack & normalization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,1,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  print(f"Post transformation size: {out.size()}")
  return out

"""
Function that reverts the changes of transform_rgb and returns an array of size [n_frames x 3 x H x W]
"""

def revert_transform_rgb(clip):
  clip = clip.permute(1, 2, 3, 0)
  clip = clip.contiguous().view(1, -1, clip.size(1), clip.size(2)).squeeze(0)
  clip = clip.mul_(255).add_(255).div(2)
  clip = clip.view(-1, clip.size(1), clip.size(2), 3)
  return clip.numpy()


# fps=25 is default for WLASL
def video2array(vname, input_dir=os.path.join(os.getcwd(), 'data/WLASL/WLASL_videos'), fps=25):
  H = W = 256 # default dims for WLASL
  name, ext = os.path.splitext(vname)
  video_path = os.path.join(input_dir, vname)
  # out = []
  get_no_of_frames = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0',video_path]
  n_frames = int(subprocess.check_output(get_no_of_frames))
  # pdb.set_trace()
  out = np.zeros((n_frames,H,W,3))
  cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt rgb24 -threads 1 -r {fps} pipe:'
  pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=10**8, stderr=subprocess.DEVNULL)

  # while True:
  for i in range(n_frames):
    buffer = pipe.stdout.read(H*W*3)
    if len(buffer) != H*W*3:
      break
    # out.append(np.frombuffer(buffer, np.uint8).reshape(H, W, 3))
    out[i,:,:,:] = np.frombuffer(buffer, dtype=np.uint8).reshape(H, W, 3)
    # pdb.set_trace()
  pipe.stdout.close()
  pipe.wait()
  # pdb.set_trace()
  return out

############# Data augmentation #############
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

############# Dataset class #############
class WLASLDataset(data.Dataset):

  def __init__(self, df, input_dir, seq_len=64,train=True, grayscale=False):
    super().__init__()
    self.df = df
    self.input_dir = input_dir
    self.video_names = list(df['video_id'])
    self.grayscale = grayscale
    self.seq_len = seq_len
    self.DataAugmentation = DataAugmentations()
    self.train = train
    self.vocab_size = len(set(self.df['gloss']))

  def __getitem__(self, idx):
    
    vname = str(self.video_names[idx])
    vname = "0" * (5-len(vname)) + vname + '.mp4'

    if self.grayscale:
      raise(NotImplementedError)

    else:
      if self.train:
        ipt = video2array(vname, self.input_dir) # convert video to np array
        images = transform_rgb(ipt) # convert to tensor, reshape and normalize img
        images = self.DataAugmentation.HorizontalFlip(images) # flip images horizontally with 50% prob
        images = self.DataAugmentation.RandomCrop(images) # take a random 224 x 224 crop
        images = self.DataAugmentation.RandomRotation(images) # randomly rotate image +- 5 degrees'
        
		# Check if we need to upsample
        if self.seq_len > images.size(1): 
          images = upsample(images, self.seq_len)

        elif self.seq_len < images.size(1): #downsample to reach seq_len
          images = downsample(images, self.seq_len)

      # validation/test dataset
      else:
        ipt = video2array(vname, self.input_dir)
        images = transform_rgb(ipt)
        images = self.DataAugmentation.HorizontalFlip(images) # flip images horizontally wiyh 50% prob
        images = self.DataAugmentation.CenterCrop(images) # center crop 224 x 224
        
        if images.size(1) <= 12:
          images = upsample(images, 13)

    # pdb.set_trace()
    # make a one-hot vector for target class
    trg = torch.zeros(self.vocab_size) # 2000 unique words
    gloss_idx = self.df.iloc[idx]['label']
    trg[gloss_idx] = 1
    
    ### Sanity checks
    #print(f"Video_id via video_names: {self.video_names[idx]}")
    #print(f"Ground truth gloss in df {self.df.iloc[idx]['label']}")
    #print(f"video_id via df {self.df.iloc[idx]['video_id']}")
    #print(f"index where target is one (1-hot vec) {np.where(trg ==1)[0][0]}")

    return images, trg
  
  def __len__(self):
    return len(self.df)


"""

################# Test up/downsampling, flipping and cropping #################
import pandas as pd
#df = pd.read_csv("/work3/s204503/bach-data/WLASL/WLASL_labels.csv")
#ipt_dir = "/work3/s204503/bach-data/WLASL/WLASL2000"
df = pd.read_csv("data/WLASL/WLASL_labels.csv")
ipt_dir = "data/WLASL/WLASL_videos"

# pdb.set_trace()
WLASL = WLASLDataset(df, ipt_dir, seq_len=64,train=True, grayscale=False)
img1, trg_word = WLASL.__getitem__(8) # example of downsampling 72 --> 64
print("FINAL SHAPE: ", img1.size())

img2, trg_word = WLASL.__getitem__(3) # example of upsampling 56 ---> 64
print(f"img2: {img2.size()}")

img1_r = revert_transform_rgb(img1)
imgs1_r = [Image.fromarray(img.astype(np.uint8)) for img in img1_r]

imgs1_r[0].show()
imgs1_r[20].show()
imgs1_r[40].show()
imgs1_r[60].show()
"""


"""
Old class where images are assumed to be in a folder for __getitem__ instead of being loaded in and converted

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
"""
