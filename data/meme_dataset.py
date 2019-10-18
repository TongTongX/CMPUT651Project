from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

VALID_LABELS = {
    'humour': ['not_funny', 'funny', 'very_funny', 'hilarious'],
    'sarcasm': ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted'],
    'offensive':
      ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive'],
    'motivational': ['not_motivational', 'motivational'],
    'overall_sentiment':
      ['very_positive', 'positive', 'neutral', 'negative', 'very_negative'],
}

class MemeDataset(Dataset):
  """Meme dataset."""
  def __init__(self, csv_file, image_dir, transform=None, is_test=False):
    """
    Args:
      csv_file (string): Path to the csv file with annotations.
      image_dir (string): Directory with all the images.
      transform (callable, optional): Optional transform to be applied
          on a sample.
    """
    self.csv_file = csv_file
    self.image_dir = image_dir
    self.transform = transform
    self.is_test = is_test
    self._preprocess_dataset()
    
  def _preprocess_dataset(self):
    self.meme_frame = pd.read_csv(filepath_or_buffer=self.csv_file)
    # Convert column names to lower case.
    self.meme_frame.columns = map(str.lower, self.meme_frame.columns)
    # Drop rows with invalid values.
    frame_len_before_drop = len(self.meme_frame)
    invalid_row_indices = self.meme_frame[
      ~self.meme_frame['humour'].isin(VALID_LABELS['humour'])
      | ~self.meme_frame['sarcasm'].isin(VALID_LABELS['sarcasm'])
      | ~self.meme_frame['offensive'].isin(VALID_LABELS['offensive'])
      | ~self.meme_frame['motivational'].isin(VALID_LABELS['motivational'])
      | ~self.meme_frame['overall_sentiment'].isin(
        VALID_LABELS['overall_sentiment'])].index
    # invalid_url_row_indices = self.get_invalid_url_row_indices()
    # invalid_row_indices = invalid_row_indices.append(invalid_url_row_indices)
    invalid_row_indices = invalid_row_indices.drop_duplicates()
    invalid_len = len(invalid_row_indices)
    print('len(invalid_row_indices): {}'.format(invalid_len))
    self.meme_frame.drop(index=invalid_row_indices, inplace=True)
    assert frame_len_before_drop - invalid_len == len(self.meme_frame)
    # Add onehot label columns
    for label in VALID_LABELS.keys():
      self.meme_frame[label + '_onehot'] = pd.get_dummies(
        data=self.meme_frame[label]).values.tolist()

  def get_invalid_url_row_indices(self):
    valid_count = 0
    invalid_indices = []
    for i, img_url in enumerate(self.meme_frame['image_url']):
      try:
        response = requests.get(url=img_url, timeout=5)
        if response.status_code != 200:
          invalid_indices.append(i)
          print('{} failed url: {}'.format(i, img_url))
          continue
        response.raise_for_status()
      except Exception as err:
        invalid_indices.append(i)
        print(i, err)
        continue
      try:
        img_from_url = np.array(Image.open(BytesIO(response.content)))
      except OSError as err:
        invalid_indices.append(i)
        print(i, err)
        continue
      # if len(img_from_url.shape) != 3 or img_from_url.shape[2] !=3:
      #   print(i, img_from_url.shape, img_url)
      #   continue
      if i % 50 == 0:
        print(i)
      valid_count += 1
    print('valid_count: {}'.format(valid_count))
    return pd.Index(invalid_indices)

  def __len__(self):
    return len(self.meme_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image_name = self.meme_frame['image_name'][idx]
    # TODO(xutong) Retrieve image from local directory
    image_path = os.path.join(self.image_dir, image_name)
    image = Image.open(fp=image_path)
    # Retrieve image from url - invalid url around 1/5 of trial dataset
    # image_url = self.meme_frame['image_url'][idx]
    # response = requests.get(url=image_url)
    # assert response.status_code == 200
    # image = Image.open(BytesIO(response.content))
    # Convert image to RGB. This drops the opacity channel if it exists.
    image = image.convert(mode='RGB')

    # Get textual input.
    ocr_extracted_text = self.meme_frame['ocr_extracted_text'][idx]
    corrected_text = self.meme_frame['corrected_text'][idx]

    # Get labels.
    humour_onehot = None
    sarcasm_onehot = None
    offensive_onehot = None
    motivational_onehot = None
    overall_sentiment_onehot = None
    if not self.is_test:
      humour_onehot = torch.from_numpy(
          np.array(self.meme_frame['humour_onehot'][idx])).unsqueeze_(0)
      sarcasm_onehot = torch.from_numpy(
          np.array(self.meme_frame['sarcasm_onehot'][idx])).unsqueeze_(0)
      offensive_onehot = torch.from_numpy(
          np.array(self.meme_frame['offensive_onehot'][idx])).unsqueeze_(0)
      motivational_onehot = torch.from_numpy(
          np.array(self.meme_frame['motivational_onehot'][idx])).unsqueeze_(0)
      overall_sentiment_onehot = (
          torch.from_numpy(np.array(
            self.meme_frame['overall_sentiment_onehot'][idx])).unsqueeze_(0))
      assert humour_onehot.shape == torch.Size([1, 4])
      assert sarcasm_onehot.shape == torch.Size([1, 4])
      assert offensive_onehot.shape == torch.Size([1, 4])
      assert motivational_onehot.shape == torch.Size([1, 2])
      assert overall_sentiment_onehot.shape == torch.Size([1, 5]) 

    sample = {
        'image_name': image_name, # For debugging.
        'image': image,
        'ocr_extracted_text': ocr_extracted_text,
        'corrected_text': corrected_text,
        'humour_onehot': humour_onehot,
        'sarcasm_onehot': sarcasm_onehot,
        'offensive_onehot': offensive_onehot,
        'motivational_onehot': motivational_onehot,
        'overall_sentiment_onehot': overall_sentiment_onehot,
        }

    if self.transform:
        sample = self.transform(sample)

    return sample

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