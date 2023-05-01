##### Dataset class for Phoenix #####
import os
from train_datasets.preprocess_PHOENIX import preprocess_df
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import pdb
from math import ceil
import random, numbers
from typing import Optional, List, Union, Tuple

def normalize(video):
  ''' stack & normalization '''
  # video = video.sub_(255).div(255)
  video = video.div(255)
  return video

def reshape(video):
  out = video.permute(1,0,2,3)
  return out

def revert_transform_rgb(clip):
  clip = clip.permute(1, 2, 3, 0)
  clip = clip.contiguous().view(1, -1, clip.size(1), clip.size(2)).squeeze(0)
  # clip = clip.mul_(255).add_(255).div(2)
  clip = clip.mul_(255)
  clip = clip.view(-1, clip.size(1), clip.size(2), 3)
  return clip.numpy()


class ColorJitter(torch.nn.Module):
    """
    Original code taken from torchvision:
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#ColorJitter.forward

    Randomly change the brightness, contrast, saturation and hue of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0) -> None:
        super().__init__()
        
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]]) -> Tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.
        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """

        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h


    def forward(self, imgs):
      
      T, C, H, W = imgs.shape

      fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
          self.brightness, self.contrast, self.saturation, self.hue)
      result = torch.empty(T, C, H, W)
      for i, img in enumerate(imgs):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        result[i] = img
        #result.append(img)
      
      #result = torch.concat(result, dim=0).view(-1, C, H, W)      
      return result


############# Data augmentations #############
class DataAugmentations:
  def __init__(self, split_type=None):
    self.H_upsample = 298
    self.W_upsample = 240
    self.H_out = 224
    self.W_out = 224
    self.split_type = split_type
    self._center_crop = torchvision.transforms.CenterCrop((self.H_out, self.W_out))
    self._center_crop_pre_random = torchvision.transforms.CenterCrop((240, 240))
    self._random_crop = torchvision.transforms.RandomCrop((self.H_out, self.W_out), padding = 0, padding_mode='constant')
    self._random_rotate = torchvision.transforms.RandomRotation(5, expand=False, fill=0.0, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
    self._upsample_pixels = torch.nn.Upsample(size=(self.H_upsample, self.W_upsample), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)
    self._color_jitter = ColorJitter(0.4, 0.4, 0.4, 0.1)
    self._to_PIL = torchvision.transforms.ToPILImage()
    self._to_tensor = torchvision.transforms.ToTensor()

  def __call__(self, vid):
    if self.split_type == 'train':
      vid = self.UpsamplePixels(vid)
      vid = normalize(vid)
      vid = self.ColorJitter(vid)
      vid = reshape(vid)
      vid = self._center_crop_pre_random(vid)
      #vid = self.HorizontalFlip(vid)
      vid = self.RandomRotation(vid)
      vid = self.RandomCrop(vid)
      
    else:
      vid = self.UpsamplePixels(vid)
      vid = normalize(vid)
      vid = reshape(vid)
      vid = self.CenterCrop(vid)
      
    return vid
  
  def UpsamplePixels(self, imgs: np.ndarray):
    return self._upsample_pixels(torch.from_numpy(imgs).double().permute(0, 3, 1, 2).contiguous()) # upsample and place color channel as dim 1
    # return imgs.permute(0, 2, 3, 1) # return and revert dim changes

  # flip all images in video horizontally with 50% probability
  def HorizontalFlip(self, imgs):
    if random.random() <= 0.5:
      imgs = torchvision.transforms.functional.hflip(imgs)
    return imgs
  
  # random 224 x 224 crop (same crop for all images in video)
  def RandomCrop(self, imgs):
    return self._random_crop(self._center_crop_pre_random(imgs))
  
  # 224 x 224 center crop (all images in video)
  def CenterCrop(self, imgs):
    return self._center_crop(imgs)
  
  # randomly rotate all images in video with +- 5 degrees.
  def RandomRotation(self, imgs):
    return self._random_rotate(imgs)

  def ColorJitter(self, imgs):
    if random.random() < 0.3:
      imgs = self._color_jitter(imgs)
    return imgs
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

# pad using replication strategy
def pad(images, seq_len):
  padding = images[:,-1,:,:].unsqueeze(1) 
  padding = torch.tile(padding, [1, seq_len-images.size(1), 1, 1]) 
  padded_images = torch.cat([images, padding], dim=1)
  return padded_images

def get_selected_indexs(input_len, t_min=0.5, t_max=1.5, max_num_frames=400):
    if t_min==1 and t_max==1:
        if input_len <= max_num_frames:
            frame_index = np.arange(input_len)
            valid_len = input_len
        else:
            sequence = np.arange(input_len)
            an = (input_len - max_num_frames)//2
            en = input_len - max_num_frames - an
            frame_index = sequence[an: -en]
            valid_len = max_num_frames
        
        if (valid_len % 4) != 0:
            valid_len -= (valid_len % 4)
            frame_index = frame_index[:valid_len]

        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len
    else:
      min_len = int(t_min*input_len)
      max_len = min(max_num_frames, int(t_max*input_len))
      output_len = np.random.randint(min_len, max_len+1)
      output_len += (4-(output_len%4)) if (output_len%4) != 0 else 0
      if input_len>=output_len: 
          selected_index = sorted(np.random.permutation(np.arange(input_len))[:output_len])
      else: 
          copied_index = np.random.randint(0,input_len,output_len-input_len)
          selected_index = sorted(np.concatenate([np.arange(input_len), copied_index]))
      assert len(selected_index) <= max_num_frames, "output_len is larger than max_num_frames"
      # pdb.set_trace()
      return selected_index, len(selected_index)


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


class PhoenixDataset(data.Dataset):
    def __init__(self, df, ipt_dir, vocab_size, split='train'):
        super().__init__()

        self.ipt_dir = ipt_dir
        self.split=split
        self.vocab_size = vocab_size
        self.df = preprocess_df(df, split, save=False, save_name=None)
        self.video_folders = list(self.df['name'])

    def __getitem__(self, idx):
        ### Assumes that within a sample (id column in df) there is only one folder named '1' ###
        image_folder = os.path.join(self.ipt_dir, self.split, self.video_folders[idx])
        # images = load_imgs(image_folder)
        image_names = np.sort(os.listdir(image_folder))
        image_names = [os.path.join(image_folder,img_name) for img_name in image_names]
        N = len(image_names)
        ipt_len = N

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
  image_path_lists, vid_lens, trgs,  trg_lens = [list(x) for x in list(zip(*data))] # ! Might be inefficient!

  if data_augmentation.split_type == "train":
    for i,image_paths in enumerate(image_path_lists):
        selected_indexs, new_len = get_selected_indexs(vid_lens[i], t_min=0.5, t_max=1.5, max_num_frames=400)
        vid_lens[i] = new_len
        image_path_lists[i] = [image_paths[idx] for idx in selected_indexs]

  max_ipt_len = max(vid_lens)
  max_trg_len = max(trg_lens)

  batch = torch.zeros((len(image_path_lists), 3, max_ipt_len, 224, 224))
  targets = torch.zeros((len(trgs), max_trg_len))
  
  for i, image_paths in enumerate(image_path_lists):
    vid = np.empty((len(image_paths), 260, 210, 3))
    for j,ipt in enumerate(image_paths):
        vid[j,:,:,:] = np.asarray(Image.open(ipt))
    vid = data_augmentation(vid)
    if vid.size(1) < max_ipt_len:
      batch[i] = pad(vid, max_ipt_len)
    else:
      batch[i] = vid
    trg_pad = torch.nn.ConstantPad1d((0, max_trg_len - len(trgs[i])), value=0)
    targets[i] = trg_pad(trgs[i])
  
  return batch, torch.tensor(vid_lens, dtype=torch.long), targets, torch.tensor(trg_lens, dtype=torch.int32)



"""
from torch.utils.data import DataLoader

class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

dp = DataPaths()
test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')[:4]
train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')[:4]
PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=1085, split='test')
PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=1085, split='train')


test_augmentations = DataAugmentations(split_type='val')
dataloaderTest = DataLoader(PhoenixTest, batch_size=1, 
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=lambda data: collator(data, test_augmentations)
                                   )

train_augmentations = DataAugmentations(split_type='train')
dataloaderTrain = DataLoader(PhoenixTrain, batch_size=4, 
                                   shuffle=True,
                                   num_workers=0,
                                   collate_fn=lambda data: collator(data, train_augmentations)
                                   )
                              
if __name__ == '__main__':
  import cv2

  for (ipt, ipt_len, trg, trg_len) in dataloaderTest:
    # pdb.set_trace()
    ipt_np = revert_transform_rgb(ipt[0])
    w = h = 224
    c = 3
    fps = 25
    sec = 10
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (w, h))
    for frame_count in range(len(ipt_np)):
      # img_ = np.random.randint(0,255, (h,w,c), dtype = np.uint8)
      img = ipt_np[frame_count].astype(np.uint8)
      # pdb.set_trace()
      video.write(img)
    video.release()

    print("IPTT", ipt.size())
    print(ipt_len)
    print("TRGG", trg.size())
    print(trg_len)
    break

  for (ipt, ipt_len, trg, trg_len) in dataloaderTrain:
    # pdb.set_trace()

    for i in range(4):
      print(f"INPUT SIZE BATCH {ipt.shape}")
      ipt_np = revert_transform_rgb(ipt[i])
      w = h = 224
      c = 3
      fps = 25
      sec = 10
      
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      # fourcc = cv2.VideoWriter_fourcc(*"avc1")
      video = cv2.VideoWriter(f'train{i}.mp4', fourcc, float(fps), (w, h))
      for frame_count in range(len(ipt_np)):
        # img_ = np.random.randint(0,255, (h,w,c), dtype = np.uint8)
        img = ipt_np[frame_count].astype(np.uint8)
        # pdb.set_trace()
        video.write(img)
      video.release()

      print("IPTT", ipt.size())
      print(ipt_len)
      print("TRGG", trg.size())
      print(trg_len)

    break 
"""
