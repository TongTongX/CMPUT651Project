from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from data.meme_dataset import MemeDataset

def main():
  trial_meme_dataset = MemeDataset(csv_file='data1.csv',
    image_dir='/home/xutong/Downloads/semeval-2020_trialdata/Meme_images/')

  fig = plt.figure()

  for i in range(len(trial_meme_dataset)):
    sample = trial_meme_dataset[i]
    print(i, np.array(sample['image']).shape, sample['corrected_text'])
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])

    if i == 3:
      plt.show()
      break

if __name__ == '__main__':
  main()