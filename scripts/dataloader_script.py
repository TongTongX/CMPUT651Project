from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

def show_batch(sample_batched):
  """Show image for a batch of samples."""
  images_batch = sample_batched['image']
  batch_size = len(images_batch)
  grid = utils.make_grid(images_batch)
  plt.imshow(grid.numpy().transpose((1, 2, 0)))
  plt.title('Batch from dataloader')

def main():
  trial_meme_dataset_transformed = MemeDataset(
    csv_file=os.path.join(os.getcwd(), 'data/data1.csv'),
    image_dir=os.path.join(os.path.expanduser('~'),
      'Downloads/semeval-2020_trialdata/Meme_images/'),
    transform=transforms.Compose(
      [ResizeSample(size=(256, 256)),
      ToTensorSample(),
      NormalizeSample(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])]))
  
  for i in range(len(trial_meme_dataset_transformed)):
    sample = trial_meme_dataset_transformed[i]
    print(i, sample['image'].size())
    if i == 3:
      break

  dataloader = DataLoader(dataset=trial_meme_dataset_transformed, batch_size=4,
    shuffle=True, num_workers=4)

  for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
      sample_batched['image'].numpy().shape)
    print('sample_batched[\'image_name\']:\n{}'.format(
      sample_batched['image_name']))
    print('sample_batched[\'humour_onehot\']:\n{}'.format(
      sample_batched['humour_onehot']))
    print('sample_batched[\'humour_int\']:\n{}'.format(
      sample_batched['humour_int']))
    print('sample_batched[\'offensive_onehot\']:\n{}'.format(
      sample_batched['offensive_onehot']))
    print('sample_batched[\'offensive_int\']:\n{}'.format(
      sample_batched['offensive_int']))
    print('sample_batched[\'ocr_extracted_text\']:\n{}'.format(
      sample_batched['ocr_extracted_text']))
    print('sample_batched[\'corrected_text\']:\n{}\n'.format(
      sample_batched['corrected_text']))

    # observe 4th batch and stop.
    if i_batch == 3:
      plt.figure()
      show_batch(sample_batched)
      plt.axis('off')
      plt.ioff()
      plt.show()
      break

if __name__ == '__main__':
  main()