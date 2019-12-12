from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

def main():
    data_transform = transforms.Compose([
            ResizeSample(size=(256, 256)),
            ToTensorSample(),
            NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    trial_meme_dataset = MemeDataset(
        csv_file=os.path.join(os.getcwd(), 'data/data1.csv'),
        image_dir=os.path.join(os.path.expanduser('~'),
        'Downloads/semeval-2020_trialdata/Meme_images/'))
    for i in range(len(trial_meme_dataset)):
        sample = trial_meme_dataset[i]
        print(i, np.array(sample['image']).shape)
        print('np.array(sample[\'image\'])[128,128,0]: {}'.format(np.array(sample['image'])[128,128,0]))
        transformed_sample = data_transform(sample)
        print(i, np.array(transformed_sample['image']).shape)
        # print(transformed_sample['image'].numpy().max(axis=1))
        print('transformed_sample[\'image\'].numpy()[0,128,128]: {}'.format(transformed_sample['image'].numpy()[0,128,128]))
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