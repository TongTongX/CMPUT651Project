from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, utils

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

from models.deep_sentiment import DeepSentimentModel
from models.deep_sentiment_vanilla import DeepSentimentVanillaModel
from models.shallownet_glove import ShallownetGloveModel
from models.model_utils import *

def show_batch(sample_batch):
    """Show image for a batch of samples."""
    images_batch = sample_batch['image']
    batch_size = len(images_batch)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

def make_weights_for_balanced_classes(meme_dataset, num_classes):
    counts = [0] * num_classes
    for i in range(len(meme_dataset)):
        counts[meme_dataset[i]['overall_sentiment_ternary_int']] += 1
    weight_per_class = [0.] * num_classes
    total_count = float(sum(counts))
    for i in range(num_classes):
        weight_per_class[i] = total_count/float(counts[i])
    sample_weights = [0] * len(meme_dataset)
    for i in range(len(meme_dataset)):
        sample_weights[i] = weight_per_class[meme_dataset[i]['overall_sentiment_ternary_int']]
    sample_weights = torch.DoubleTensor(sample_weights)
    return sample_weights

def main():
    # Create training and validation datasets
    print("Initializing Datasets and Dataloaders...")
    trial_meme_dataset_transformed = MemeDataset(
        csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
        image_dir=os.path.join(os.getcwd(),
        '../data/semeval-2020_trialdata/Meme_images/'),
        transform=transforms.Compose(
            [
                ResizeSample(size=(299, 299)),  # For Inception
                # ResizeSample(size=(224, 224)),  # For other pretrained models
                ToTensorSample(),
                NormalizeSample(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])]))

    trial_meme_train, trial_meme_val = random_split(
        dataset=trial_meme_dataset_transformed, lengths=[800, 200])

    # Create training and validation dataloaders
    # Balanced class============================================================
    # sample_weights_train = make_weights_for_balanced_classes(
    #     trial_meme_train, num_classes=3)
    # weighted_sampler_train = WeightedRandomSampler(
    #     sample_weights_train, len(sample_weights_train))
    # train_dataloader = DataLoader(dataset=trial_meme_train, batch_size=4,
    #     sampler=weighted_sampler_train, num_workers=4)
  
    # sample_weights_val = make_weights_for_balanced_classes(
    #     trial_meme_val, num_classes=3)
    # weighted_sampler_val = WeightedRandomSampler(
    #     sample_weights_val, len(sample_weights_val))
    # val_dataloader = DataLoader(dataset=trial_meme_val, batch_size=4,
    #     sampler=weighted_sampler_val, num_workers=4)
    # ==========================================================================

    # Imbalanced class==========================================================
    train_dataloader = DataLoader(dataset=trial_meme_train, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=trial_meme_val, batch_size=4, shuffle=True, num_workers=4)
    # ==========================================================================

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    deepsent_config = {
        'num_classes': 3, # negative, positive, neutral
        'batch_size': 4, 'vocab_size': 400000, 'embedding_dim': 300}
    deepsent = DeepSentimentModel(**deepsent_config)
    # deepsent = DeepSentimentVanillaModel(**deepsent_config)
    # deepsent = ShallownetGloveModel(**deepsent_config)
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
        criterion=criterion, optimizer=optimizer_ft, num_epochs=10,
        is_inception=True, target_label='overall_sentiment_ternary_int')

if __name__ == '__main__':
    main()