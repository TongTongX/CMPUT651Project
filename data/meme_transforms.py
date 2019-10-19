from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
