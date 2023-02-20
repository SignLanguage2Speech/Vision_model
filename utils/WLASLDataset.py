import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import subprocess
import shlex
import torchvision

"""
Function that converts List of RGB images to [batch_size x 3 x n_frames x H x W]
"""
def transform_rgb(snippet):
  #print(f"Number of images being transformed: {len(snippet)}")
  #print(f"Original image size: {snippet[0].shape}")
  ''' stack & normalization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  #print(f"Post transformation size: {out.size()}")
  #[333, 333, 3] --> [1, 3, 21, 333, 333]
  return out

"""
Function that converts List of grayscale images to [batch_size x 1 x n_frames x H x W]
"""
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

"""
Function that reverts the changes of transform_rgb and returns an array of size [n_frames x 3 x H x W]
"""

def revert_transform_rgb(clip):
  clip = clip.squeeze(0).permute(1, 2, 3, 0)
  clip = clip.contiguous().view(1, -1, clip.size(1), clip.size(2)).squeeze(0)
  clip = clip.mul_(255).add_(255).div(2)
  clip = clip.view(-1, clip.size(1), clip.size(2), 3)
  return clip.numpy()


# fps=25 is default for WLASL
def video2array(vname, input_dir=os.path.join(os.getcwd(), 'data/WLASL/WLASL_videos'), fps=25):
  H = W = 256 # default dims for WLASL
  name, ext = os.path.splitext(vname)
  video_path = os.path.join(input_dir, vname)
  out = []
  subprocess.run(shlex.split(f'ffmpeg -y -f lavfi -i testsrc=size={W}x{H}:rate=1 -vcodec libx264 -g 20 -crf 17 -pix_fmt yuv420p -t 6000 {video_path}'), stderr=subprocess.DEVNULL)
  
  # the cmd below creates more frames due to the -qscale:v 0 argument... decide which we want we want to use 
  #cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt bgr24 -threads 1 -r {fps} -vf scale=-1:331 -qscale:v 0 pipe:' 
  
  cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt rgb24 -threads 1 -r {fps} pipe:'
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=10**8, stderr=subprocess.DEVNULL)
  while True:
    buffer = process.stdout.read(H*W*3)
    if len(buffer) != H*W*3:
      break
    out.append(np.frombuffer(buffer, np.uint8).reshape(H, W, 3))

  process.stdout.close()
  process.wait()
  return out

############# Data augmentation #############
class DataAugmentations:
  def __init__(self):
    self.H_out = 224
    self.W_out = 224

  def HorizontalFlip(self, imgs):
    p = np.random.randint(0, 2)
    if p == 0:
      # flip all images in video horizontally
      imgs = [np.flip(e, axis=1) for e in imgs]
    return imgs

  def RandomCrop(self, imgs):
    crop = torchvision.transforms.RandomCrop((self.H_out, self.W_out), padding = 0, padding_mode='constant')
    return crop(imgs)
  
  def CenterCrop(self, imgs):
    crop = torchvision.transforms.CenterCrop((self.H_out, self.W_out))
    return crop(imgs)
  
  def RandomRotation(self, imgs):
    rotate = torchvision.transforms.RandomRotation(5, expand=False, fill=0)
    return rotate(imgs)

def upsample(images, seq_len):
  images_org = images.detach().clone() # create a clone of original input
  if seq_len / images.size(2) >= 2: # check if image needs to be duplicated
    repeats = int(np.floor(seq_len / images.size(2)) - 1) # number of concats
    for _ in range(repeats):
      images = torch.cat((images, images_org), dim=2) # concatenate images temporally
  
  if seq_len > images.size(2):
    start_idx = np.random.randint(0, images_org.size(2) - (seq_len - images.size(2))) 
    stop_idx = start_idx + (seq_len - images.size(2))
    images = torch.cat((images, images_org[:, :, start_idx:stop_idx]), dim=2)
  
  return images

def downsample(images, seq_len):
  start_idx = np.random.randint(0, images.size(2) - seq_len)
  stop_idx = start_idx + seq_len
  return images[:, :, start_idx:stop_idx]


############# Dataset class #############
class WLASLDataset(data.Dataset):

  def __init__(self, df, input_dir, seq_len=64,train=True, grayscale=False):
    super().__init__()
    self.df = df
    self.input_dir = input_dir
    self.video_names = os.listdir(self.input_dir)
    self.grayscale = grayscale
    self.seq_len = seq_len
    self.DataAugmentation = DataAugmentations
    self.train = train

  def __getitem__(self, idx):

    if self.grayscale:
      raise(NotImplementedError)
    else:
      if self.train:
        ipt = video2array(self.video_names[idx], self.input_dir)
        ipt = self.DataAugmentation.HorizontalFlip(ipt) # flip images horizontally wiyh 50% prob
        images = transform_rgb(ipt)
        images = self.DataAugmentation.RandomCrop(images) # take a random 224 x 224 crop
      else:
        ipt = video2array(self.video_names[idx], self.input_dir)
        ipt = self.DataAugmentation.HorizontalFlip(ipt) # flip images horizontally wiyh 50% prob
        images = transform_rgb(ipt)
        images = self.DataAugmentation.CenterCrop(images) # center crop 224 x 224
        
  
    # Check if we need to upsample
    if self.seq_len > images.size(2): 
      images = upsample(images, self.seq_len)

    # Check if we need to downsample
    elif self.seq_len < images.size(2): #downsample to reach seq_len
      images = downsample(images, self.seq_len)

    trg = self.df['gloss'][idx]
    #trg = self.df['label'][idx]
    return images, trg
  
  def __len__(self):
    return len(self.video_names)
      

"""

################# Test up/downsampling, flipping and cropping #################
import pandas as pd
df = pd.read_csv('data/WLASL/WLASL_labels.csv')
img_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_videos')
WLASL = WLASLTrainDataset(df, img_folder, seq_len=64, grayscale=False)
img1, trg_word = WLASL.__getitem__(8) # example of downsampling 72 --> 64
print("FINAL SHAPE: ", img1.size())

img2, trg_word = WLASL.__getitem__(3) # example of upsampling 56 ---> 64
print(f"img2: {img2.size()}")

img1_r = revert_transform_rgb(img1)
imgs1_r = [Image.fromarray(img.astype(np.uint8)) for img in img1_r]
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
