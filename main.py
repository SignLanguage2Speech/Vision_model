import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from model import S3D
from utils.load_weigths import load_model_weights
from PIL import Image

#import subprocess
#subprocess.call('pip list')
### opencv-python is here but cv2 gives importerror...

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

images = {video_id : [] for video_id in video_names}
for v in video_names[:1]:
  video_path = os.path.join(img_folder, v)
  #imgs = [np.expand_dims(np.asarray(Image.open(os.path.join(video_path, f)).convert('L')),2) for f in os.listdir(video_path)[1:]]
  #print(imgs[0].shape)
  images[v] = transform_gray([np.expand_dims(np.asarray(Image.open(os.path.join(video_path, f)).convert('L')),2) for f in os.listdir(video_path)])
  #images[v] = transform([np.asarray(Image.open(os.path.join(video_path, f))) for f in os.listdir(video_path)[1:]])

class WLASLDataset(data.Dataset):
  def __init__(self, df, img_folder, gray_scale=False):
    super().__init__()
    self.df = df
    self.img_folder = img_folder
    self.video_names = os.listdir(self.img_folder)
    self.gray_scale = gray_scale

  def __getitem__(self, idx):
    video_path = os.path.join(self.img_folder, self.video_names[idx])
    
    if self.gray_scale:
      images = transform_gray([np.expand_dims(np.asarray(Image.open(os.path.join(video_path, f)).convert('L')),2) for f in os.listdir(video_path)])
    else:
      images = transform_rgb([np.asarray(Image.open(os.path.join(video_path, f))) for f in os.listdir(video_path)])

    trg = self.df['label']

    return images, trg
  
  def __len__(self):
    return len(self.video_names)

def main():
  # load data
  df = pd.read_csv('data/WLASL/WLASL_labels.csv')
  img_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_images')
  video_names = os.listdir(img_folder)
  images = {video_id : [] for video_id in video_names}
  # load model
  n_classes = len(set(df['gloss'])) #2000
  model = S3D(n_classes)
  model = load_model_weights(model, 'S3D_kinetics400.pt')
  
  for v in video_names[:10]:
    video_path = os.path.join(img_folder, v)
    images[v] = transform([np.asarray(Image.open(os.path.join(video_path, f))) for f in os.listdir(video_path)])
  
  out = model.forward(images[video_names[0]])
  print(torch.softmax(out, dim=1).size())
  print(torch.sum(torch.softmax(out, dim=1)))
  pred = torch.argmin(torch.log_softmax(out, dim=1))
  
  criterion = nn.CrossEntropyLoss()

  #loss = criterion()



#if __name__ == '__main__':
  #main()