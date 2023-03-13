import os
import numpy as np
import pandas as pd
"""
annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
test = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')
val = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
attributeNames = list(train.columns)

print(attributeNames)
print(train.head(5))
print(os.listdir(features_path))

chars = '?.,!-_+'

annotations = list(train['orth'])# + list(test['orth']) + list(val['orth'])
annotations = list(sorted(set([word for sent in annotations for word in sent.replace(chars,'').split(' ')])))
print(f"Train vocab size: {len(annotations)}")

"""
from PIL import Image
import torch

### Testing Dataset class locally ###
ipt_dir = os.path.join(os.getcwd(), 'PHOENIX\\sample_data')

def load_imgs(ipt_dir):
  image_names = os.listdir(ipt_dir)
  N = len(image_names)
  imgs = np.empty((N, 260, 210, 3))

  for i in range(N):
    imgs[i,:,:,:] = np.asarray(Image.open(os.path.join(ipt_dir, image_names[i])))
  
  return imgs

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


video_names = os.listdir(ipt_dir)
imgs = load_imgs(os.path.join(ipt_dir, video_names[0]))
print(f"loaded size {imgs.shape}")
Image.fromarray(imgs[0].astype(np.uint8)).show()

#imgs = transform_rgb(imgs)
#imgs = revert_transform_rgb(imgs)
#Image.fromarray(imgs[0].astype(np.uint8)).show()


Upsample = torch.nn.Upsample(size=(298, 240), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)

imgs = torch.from_numpy(imgs).double().permute(0, 3, 1, 2).contiguous()
print(f"PRE CONV SIZE {imgs.size()}")

imgs = Upsample(imgs)


imgs2 = imgs.permute(0, 2, 3, 1)

print("POST PERMUTE SIZE: ", imgs2.size())
imgs2 = transform_rgb(imgs2.numpy())
imgs2 = revert_transform_rgb(imgs2)
Image.fromarray(imgs2[0].astype(np.uint8)).show()

imgs2= transform_rgb(imgs2)
print(f"Post transform {imgs2.size()}")

"""
class DataAugmentations:
  def __init__(self):
    self.H_upsample = 298
    self.W_upsample = 240
    self.H_out = 224
    self.W_out = 224

  def UpsamplePixels(self, imgs: np.ndarray):
    Upsample = torch.nn.Upsample(size=(self.H_upsample, self.W_upsample), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)
    imgs = Upsample(torch.from_numpy(imgs).double().permute(0, 3, 1, 2).contiguous()) # upsample and place color channel as dim 1
    return imgs.permute(0, 2, 3, 1) # return and revert dim changes

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
"""



#imgs = TransposeConv(imgs)
#print(imgs.shape)
### train augmentations ###

