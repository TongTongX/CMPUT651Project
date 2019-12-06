import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils

import os
import numpy as np
import matplotlib.pyplot as plt

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

from models.deep_sentiment_att import DeepSentimentAttentionModel
from models.deep_sentiment_fusion import DeepSentimentFusionModel
from models.deep_sentiment_svm import DeepSentimentSVM_Model
from models.incept_roberta_att import InceptRobertaAttentionModel
from models.model_utils import *

from sklearn.svm import SVC

import time
import copy


def get_dataloaders(data_path, img_path, batch_size, split_seq):
    # split_seq: [0.8, 0.2], 80% data for training, 10% for validation, the rest of data for testing
    data_transform = transforms.Compose([
      ResizeSample(size=(299, 299)),
      ToTensorSample(),
      NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    meme_dataset_transformed = MemeDataset(
      csv_file=os.path.join(os.getcwd(), data_path),
      image_dir = os.path.join(os.getcwd(), img_path),
        transform=data_transform)

    # Split the dataset
    train_len = int(len(meme_dataset_transformed) * split_seq[0])
    # val_len = int(len(meme_dataset_transformed) * split_seq[1])
    test_len = len(meme_dataset_transformed) - train_len

    # meme_train, meme_val, meme_test = random_split(meme_dataset_transformed, [train_len, val_len, test_len])
    meme_train, meme_val = random_split(meme_dataset_transformed, [train_len, test_len])

    # The dataloader for training, validation and testing dataset
    train_dataloader = DataLoader(dataset=meme_train, batch_size=batch_size,
        shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=meme_val, batch_size=batch_size,
        shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(dataset=meme_test, batch_size=4,
    #     shuffle=True, num_workers=4)

    # dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    return dataloaders_dict


def main():
    trial_data_path = '../data/data1.csv'
    trial_img_path = '../data/semeval-2020_trialdata/Meme_images/'

    train_data_path = '../data/data_7000_new.csv'
    train_img_path = '../data/memotion_analysis_training_data/data_7000/'

    batch_size = 64

    dataloaders_dict = get_dataloaders(train_data_path, train_img_path, batch_size, [0.8, 0.2])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameter tuning
    num_att_head = [1]

    for att_head in num_att_head:
        deepsentatt_config = {
            'num_classes': 3, # negative, positive, neutral
            'batch_size': batch_size, 'vocab_size': 400000, 'embedding_dim': 300, 'att_head_num': att_head}
        # DeepSentimentAttentionModel
        # DeepSentimentFusionModel
        # DeepSentimentSVM_Model
        # InceptRobertaAttentionModel
        deepsentatt_model = InceptRobertaAttentionModel(**deepsentatt_config)
        # Send the model to GPU
        deepsentatt_model = deepsentatt_model.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        feature_extract = True
        params_to_update = deepsentatt_model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in deepsentatt_model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in deepsentatt_model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        deepsentatt_model, hist = train_att_model(model=deepsentatt_model, dataloaders=dataloaders_dict,
            criterion=criterion, optimizer=optimizer_ft, att_head_num=att_head, num_epochs=5, is_inception=True)
        # deepsentatt_model, hist = train_svm_model(model=deepsentatt_model, dataloaders=dataloaders_dict,
        #     criterion=criterion, optimizer=optimizer_ft, num_epochs=20, is_inception=True)


main()
