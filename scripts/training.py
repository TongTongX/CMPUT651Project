from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

from models.deep_sentiment import DeepSentimentModel
from models.deep_sentiment_vanilla import DeepSentimentVanillaModel
from models.model_utils import *

def show_batch(sample_batch):
  """Show image for a batch of samples."""
  images_batch = sample_batch['image']
  batch_size = len(images_batch)
  grid = utils.make_grid(images_batch)
  plt.imshow(grid.numpy().transpose((1, 2, 0)))
  plt.title('Batch from dataloader')

def main():
  # Create training and validation datasets
  print("Initializing Datasets and Dataloaders...")
  trial_meme_dataset_transformed = MemeDataset(
    csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
    image_dir=os.path.join(os.getcwd(),
      '../data/semeval-2020_trialdata/Meme_images/'),
    transform=transforms.Compose(
      [ResizeSample(size=(224, 224)),
      ToTensorSample(),
      NormalizeSample(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])]))

  trial_meme_train, trial_meme_val = random_split(
    dataset=trial_meme_dataset_transformed, lengths=[800, 200])

  # Create training and validation dataloaders
  train_dataloader = DataLoader(dataset=trial_meme_train, batch_size=4,
    shuffle=True, num_workers=4)
  val_dataloader = DataLoader(dataset=trial_meme_val, batch_size=4,
    shuffle=True, num_workers=4)
  dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  deepsent_config = {
    'num_classes': 5, # negative, positive, neutral
    'batch_size': 4, 'vocab_size': 400000, 'embedding_dim': 300}
#   deepsent = DeepSentimentModel(**deepsent_config)
  deepsent = DeepSentimentVanillaModel(**deepsent_config)
  # Send the model to GPU
  deepsent = deepsent.to(device)

  # Gather the parameters to be optimized/updated in this run. If we are
  #  finetuning we will be updating all parameters. However, if we are
  #  doing feature extract method, we will only update the parameters
  #  that we have just initialized, i.e. the parameters with requires_grad
  #  is True.
  feature_extract = True
  params_to_update = deepsent.parameters()
  print("Params to learn:")
  if feature_extract:
    params_to_update = []
    for name,param in deepsent.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
  else:
    for name,param in deepsent.named_parameters():
      if param.requires_grad == True:
        print("\t",name)

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
  # Setup the loss fxn
  criterion = nn.CrossEntropyLoss()

  # Train and evaluate
  deepsent, hist = train_model(model=deepsent, dataloaders=dataloaders_dict,
    criterion=criterion, optimizer=optimizer_ft, num_epochs=1,
    is_inception=True, target_label='overall_sentiment_int')


'''
  for i_batch, sample_batch in enumerate(train_dataloader):
    print(i_batch, sample_batch['image'].size(),
      sample_batch['image'].numpy().shape)
    print('sample_batch[\'image_name\']:\n{}'.format(
      sample_batch['image_name']))
    print('sample_batch[\'humour_onehot\']:\n{}'.format(
      sample_batch['humour_onehot']))
    print('sample_batch[\'humour_int\']:\n{}'.format(
      sample_batch['humour_int']))
    print('sample_batch[\'offensive_onehot\']:\n{}'.format(
      sample_batch['offensive_onehot']))
    print('sample_batch[\'offensive_int\']:\n{}'.format(
      sample_batch['offensive_int']))
    print('sample_batch[\'ocr_extracted_text\']:\n{}'.format(
      sample_batch['ocr_extracted_text']))
    print('sample_batch[\'corrected_text\']:\n{}\n'.format(
      sample_batch['corrected_text']))

    image_batch = sample_batch['image']
  
    ocr_text_batch = sample_batch['ocr_extracted_text']
    corrected_text_batch = sample_batch['corrected_text']
    print('ocr_text_batch:\n{}'.format(ocr_text_batch))
    print('corrected_text_batch:\n{}'.format(corrected_text_batch))
    while 'nan' in corrected_text_batch:
      nan_idx = corrected_text_batch.index('nan')
      corrected_text_batch[nan_idx] = ocr_text_batch[nan_idx]
    print('corrected_text_batch:\n{}\n\n'.format(corrected_text_batch))

    deepsent_config = {
      'num_classes': 3, # negative, positive, neutral
      'batch_size': 4, 'vocab_size': 400000, 'embedding_dim': 300}
    deepsent = DeepSentimentModel(**deepsent_config)
    output = deepsent(image_batch=image_batch, text_batch=corrected_text_batch)
    print('output.size(): {}'.format(output.size()))
    print('output:\n{}'.format(output))
    # observe 4th batch and stop.
    if i_batch == 0:
      plt.figure()
      show_batch(sample_batch)
      plt.axis('off')
      plt.ioff()
      plt.show()
      break
'''
if __name__ == '__main__':
  main()