from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

def main():
  data_transform = transforms.Compose([
          ResizeSample(size=(256, 256)),
          ToTensorSample(),
          NormalizeSample(mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375])
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
    print('transformed_sample[\'image\'].numpy()[1,2,3]: {}'.format(
      transformed_sample['image'].numpy()[1,2,3]))
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(transformed_sample['image'].numpy().transpose((1, 2, 0)))

    if i == 3:
      plt.show()
      break

if __name__ == '__main__':
  main()