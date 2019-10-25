from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score, f1_score

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

def main():
  data_transform = transforms.Compose([
    ResizeSample(size=(256, 256)),
    ToTensorSample(),
    NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
  trial_meme_dataset_transformed = MemeDataset(
    csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
    image_dir = os.path.join(os.getcwd(), '../data/semeval-2020_trialdata/Meme_images/'),
      transform=data_transform)

  train_meme_dataset_transformed = MemeDataset(
    csv_file=os.path.join(os.getcwd(), '../data/data_7000_new.csv'),
    image_dir = os.path.join(os.getcwd(), '../data/memotion_analysis_training_data/data_7000/'),
      transform=data_transform)

  evaluate_classification(meme_dataset_transformed=trial_meme_dataset_transformed)

def evaluate_classification(meme_dataset_transformed):
  dataloader = DataLoader(dataset=meme_dataset_transformed, batch_size=1,
    shuffle=True, num_workers=0)
  # Binary classification
  humour_binary_pred = []
  sarcasm_binary_pred = []
  offensive_binary_pred = []
  motivational_binary_pred = []
  humour_binary_true = []
  sarcasm_binary_true = []
  offensive_binary_true = []
  motivational_binary_true = []
  # Ternary classification
  overall_sentiment_ternary_pred = []
  overall_sentiment_ternary_true = []
  # Multiclass classification
  humour_multiclass_pred = []
  sarcasm_multiclass_pred = []
  offensive_multiclass_pred = []
  overall_sentiment_multiclass_pred = []
  humour_multiclass_true = []
  sarcasm_multiclass_true = []
  offensive_multiclass_true = []
  overall_sentiment_multiclass_true = []

  for i_batch, sample in enumerate(dataloader):
    if i_batch % 50 == 0:
      print(i_batch)
    np.random.seed(seed=i_batch)
    # Prediction----------------------------------------------------------------
    # Binary prediction.
    humour_binary_pred.append(np.random.randint(2))
    sarcasm_binary_pred.append(np.random.randint(2))
    offensive_binary_pred.append(np.random.randint(2))
    motivational_binary_pred.append(np.random.randint(2))
    # Ternary prediction.
    overall_sentiment_ternary_pred.append(np.random.randint(3))
    # Multiclass prediction.
    humour_multiclass_pred.append(np.random.randint(4))
    sarcasm_multiclass_pred.append(np.random.randint(4))
    offensive_multiclass_pred.append(np.random.randint(4))
    overall_sentiment_multiclass_pred.append(np.random.randint(5))

    # True value----------------------------------------------------------------
    humour_int_true = sample['humour_int'].item()
    sarcasm_int_true = sample['sarcasm_int'].item()
    offensive_int_true = sample['offensive_int'].item()
    motivational_int_true = sample['motivational_int'].item()
    overall_sentiment_int_true = sample['overall_sentiment_int'].item()
    # Binary true value.
    humour_binary_true.append(0 if humour_int_true==0 else 1)
    sarcasm_binary_true.append(0 if sarcasm_int_true==0 else 1)
    offensive_binary_true.append(0 if offensive_int_true==0 else 1)
    motivational_binary_true.append(motivational_int_true)
    # Ternary true value.
    if overall_sentiment_int_true < 2:
      overall_sentiment_ternary_true.append(0)
    elif overall_sentiment_int_true > 2:
      overall_sentiment_ternary_true.append(2)
    else:
      overall_sentiment_ternary_true.append(1)
    # Multiclass true value.
    humour_multiclass_true.append(humour_int_true)
    sarcasm_multiclass_true.append(sarcasm_int_true)
    offensive_multiclass_true.append(offensive_int_true)
    overall_sentiment_multiclass_true.append(overall_sentiment_int_true)

  # accuracy_score
  print('humour binary accuracy_score: {}'.format(accuracy_score(humour_binary_true, humour_binary_pred)))
  print('sarcasm binary accuracy_score: {}'.format(accuracy_score(sarcasm_binary_true, sarcasm_binary_pred)))
  print('offensive binary accuracy_score: {}'.format(accuracy_score(offensive_binary_true, offensive_binary_pred)))
  print('motivational binary accuracy_score: {}'.format(accuracy_score(motivational_binary_true, motivational_binary_pred)))
  print('overall_sentiment ternary accuracy_score: {}'.format(accuracy_score(overall_sentiment_ternary_true, overall_sentiment_ternary_pred)))
  print('humour multiclass accuracy_score: {}'.format(accuracy_score(humour_multiclass_true, humour_multiclass_pred)))
  print('sarcasm multiclass accuracy_score: {}'.format(accuracy_score(sarcasm_multiclass_true, sarcasm_multiclass_pred)))
  print('offensive multiclass accuracy_score: {}'.format(accuracy_score(offensive_multiclass_true, offensive_multiclass_pred)))
  print('overall_sentiment multiclass accuracy_score: {}'.format(accuracy_score(overall_sentiment_multiclass_true, overall_sentiment_multiclass_pred)))
  # f1_score
  print('humour binary f1_score: {}'.format(f1_score(humour_binary_true, humour_binary_pred, average='macro')))
  print('sarcasm binary f1_score: {}'.format(f1_score(sarcasm_binary_true, sarcasm_binary_pred, average='macro')))
  print('offensive binary f1_score: {}'.format(f1_score(offensive_binary_true, offensive_binary_pred, average='macro')))
  print('motivational binary f1_score: {}'.format(f1_score(motivational_binary_true, motivational_binary_pred, average='macro')))
  print('overall_sentiment ternary f1_score: {}'.format(f1_score(overall_sentiment_ternary_true, overall_sentiment_ternary_pred, average='macro')))
  print('humour multiclass f1_score: {}'.format(f1_score(humour_multiclass_true, humour_multiclass_pred, average='macro')))
  print('sarcasm multiclass f1_score: {}'.format(f1_score(sarcasm_multiclass_true, sarcasm_multiclass_pred, average='macro')))
  print('offensive multiclass f1_score: {}'.format(f1_score(offensive_multiclass_true, offensive_multiclass_pred, average='macro')))
  print('overall_sentiment multiclass f1_score: {}'.format(f1_score(overall_sentiment_multiclass_true, overall_sentiment_multiclass_pred, average='macro')))


if __name__ == '__main__':
  main()