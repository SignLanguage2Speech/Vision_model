import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import subprocess
import shlex

"""
Function that converts List of RGB images to [batch_size x 3 x n_frames x H x W]
"""
def transform_rgb(snippet):
  print(f"Number of images being transformed: {len(snippet)}")
  print(f"Original image size: {snippet[0].shape}")
  ''' stack & normalization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  print(f"Post transformation size: {out.size()}")
  #[333, 333, 3] --> [1, 3, 21, 333, 333]
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

# fps=25 is default for WLASL
def video2array(vname, input_dir=os.path.join(os.getcwd(), 'data/WLASL_test/videos'), fps=25):
  H = W = 256 # default dims for WLASL
  name, ext = os.path.splitext(vname)
  video_path = os.path.join(input_dir, vname)
  out = []
  print(video_path)
  subprocess.run(shlex.split(f'ffmpeg -y -f lavfi -i testsrc=size={W}x{H}:rate=1 -vcodec libx264 -g 20 -crf 17 -pix_fmt yuv420p -t 6000 {video_path}'), stderr=subprocess.DEVNULL)
  
  # the cmd below creates more frames due to the -qscale:v 0 argument... decide which we want we want to use 
  #cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt bgr24 -threads 1 -r {fps} -vf scale=-1:331 -qscale:v 0 pipe:' 
  
  cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt rgb24 -threads 1 -r {fps} pipe:'
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=10**8, stderr=subprocess.DEVNULL)
  while True:
    buffer = process.stdout.read(H*W*3)
    if len(buffer) != H*W*3:
      print(f"dimension mismatch... {len(buffer)}!= {H*W*3}")
      break
    out.append(np.frombuffer(buffer, np.uint8).reshape(H, W, 3))

  process.stdout.close()
  process.wait()

  return out

"""
Test up and downsampling frames in __getitem__ next
"""

class WLASLDataset(data.Dataset):

  def __init__(self, df, input_dir, seq_len=64, grayscale=False):
    super().__init__()
    self.df = df
    self.input_dir = input_dir
    self.video_names = os.listdir(self.input_dir)
    self.grayscale = grayscale
    self.seq_len = seq_len

  def __getitem__(self, idx):
    video_path = os.path.join(self.input_dir, self.video_names[idx])
    if self.grayscale:
      raise(NotImplementedError)
    else:
      images = transform_rgb(video_path)

    # Check if we need to upsample
    if self.seq_len < images.size(2): 
      images_org = images.detach().clone()
      if self.seq_len/images.size(2) > 2: # check if image needs to be duplicated
        repeats = np.floor(self.seq_len/images.size(2)) 
        for i in range(repeats):
          images = images.cat((images, images_org), dim=2) # concatenate images temporally
      else:
        start_idx = np.random.randint(0, images.size(2)-self.seq_len)
        images = torch.cat((images, images_org[:][:][start_idx:]))
    
    # Check if we need to downsample
    elif self.seq_len > images.size(2): #downsample to reach seq_len
      start_idx = np.random.randint(0, images.size(2)-self.seq_len)
      images = images[:][:][start_idx:]

    trg = self.df['gloss'][idx]
    #trg = self.df['label'][idx]
    return images, trg
  
  def __len__(self):
    return len(self.video_names)

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
