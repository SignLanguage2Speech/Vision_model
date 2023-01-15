import os
import numpy as np
import pandas as pd
import torch
from model import S3D
from utils.load_weigths import load_model_weights
from PIL import Image


#import subprocess
#subprocess.call('pip list')
### opencv-python is here but cv2 gives importerror...


df = pd.read_csv('data/WLASL/WLASL_labels.csv')

imgs_folder = os.path.join(os.getcwd(), 'data/WLASL/WLASL_images')
video_names = os.listdir(imgs_folder)

images = {video_id : [] for video_id in video_names}

for v in video_names[:10]:
  video_path = os.path.join(imgs_folder, v)
  images[v] = [Image.open(os.path.join(video_path, f)) for f in os.listdir(video_path)]

  
    



def transform(snippet):
  print(f"Number of images being transformed: {len(snippet)}")
  print(f"Original image size: {snippet[0].shape}")
  ''' stack & noralization '''
  snippet = np.concatenate(snippet, axis=-1)
  snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
  snippet = snippet.mul_(2.).sub_(255).div(255)
  out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
  print(f"Post transformation size: {out.size()}")
  return out

def main():
    n_classes = len(set(df['gloss'])) #2000
    model = S3D(n_classes)
    model = load_model_weights(model, 'S3D_kinetics400.pt')
    
