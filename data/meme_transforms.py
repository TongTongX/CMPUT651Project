from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from data.meme_dataset import MemeDataset

class ResizeSample(transforms.Resize):
  def __init__(self, size, interpolation=Image.BILINEAR):
    super().__init__(size=size, interpolation=interpolation)

  def __call__(self, sample):
    rescaled_image = super().__call__(img=sample['image'])
    sample['image'] = rescaled_image
    return sample

  def __repr__(self):
    return super().__repr__()

class ToTensorSample(transforms.ToTensor):
  def __call__(self, sample):
    image_tensor = super().__call__(pic=sample['image'])
    sample['image'] = image_tensor
    return sample

  def __repr__(self):
    return super().__repr__()

class NormalizeSample(transforms.Normalize):
  def __init__(self, mean, std, inplace=False):
    super().__init__(mean=mean, std=std, inplace=inplace)

  def __call__(self, sample):
    try:
      normalized_tensor = super().__call__(tensor=sample['image'])
    except RuntimeError as err:
      print(sample['image_name'], sample['image'].numpy().shape)
      raise
    sample['image'] = normalized_tensor
    return sample

  def __repr__(self):
    return super().__repr__()

def main():
  data_transform = transforms.Compose([
          ResizeSample(size=(256, 256)),
          ToTensorSample(),
          NormalizeSample(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
      ])

  # Apply each of the above transforms on sample.
  fig = plt.figure()
  trial_meme_dataset = MemeDataset(
    csv_file='data1.csv',
    image_dir='/home/xutong/Downloads/semeval-2020_trialdata/Meme_images/')
  for i in range(len(trial_meme_dataset)):
    sample = trial_meme_dataset[i]
    print(i, np.array(sample['image']).shape)
    transformed_sample = data_transform(sample)
    print(i, np.array(transformed_sample['image']).shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(transformed_sample['image'].numpy().transpose((1, 2, 0)))

    if i == 3:
      plt.show()
      break