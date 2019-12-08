import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

LABEL_DICT = {
    'humour_int':0,
    'sarcasm_int':1,
    'offensive_int':2,
    'motivational_int':3,
    'overall_sentiment_int':4}

def readData(datalabel,batch_size):
    data_transform = transforms.Compose([
        ResizeSample(size=(256, 256)),
        ToTensorSample(),
        NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if datalabel == 'trial':
        dataset = MemeDataset(
            csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
            image_dir = os.path.join(os.getcwd(), '../data/semeval-2020_trialdata/Meme_images/'),
            transform= data_transform)
    else:
        dataset = MemeDataset(
            csv_file=os.path.join(os.getcwd(), '../data/data_7000_new.csv'),
            image_dir = os.path.join(os.getcwd(), '../data/memotion_analysis_training_data/data_7000/'),
            transform=data_transform)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=True, num_workers=0)
    
    return list(dataloader), len(list(dataloader))

def getBatchSample(dataloader,b_idx,datatype='all'):
    sample = dataloader[b_idx]
    if datatype == 'text':
        return sampletxt2data(sample)
    elif datatype == 'texty':
        return sampletxt2data(sample),sampley2data(sample)
    return sampletxt2data(sample),sampleimg2data(sample),sampley2data(sample)

def sampletxt2data(sample):
    txtdata = dict()
    image_name = np.asarray(sample['image_name'])
    ocr_extracted_text = np.asarray(sample['ocr_extracted_text'])
    corrected_text = np.asarray(sample['corrected_text'])

    for i in range(len(corrected_text)):
        if corrected_text[i] == " ":
            txtdata[image_name[i]] = ocr_extracted_text[i]
        else:
            txtdata[image_name[i]] = corrected_text[i]
    return txtdata

def sampleimg2data(sample):
    imgdata = dict()
    image_name = np.asarray(sample['image_name'])
    flat_img = np.asarray([np.asarray(torch.flatten(x)) for x in sample['image']])

    for i in range(len(flat_img)):
        imgdata[image_name[i]] = flat_img
    return imgdata

def sampley2data(sample, _LABEL_DICT=LABEL_DICT):
    ydata = dict()
    # image_name = np.asarray(sample['image_name'])
    # y = list() 
    # for label_type in _LABEL_DICT:
    #     y.append(np.asarray(sample[label_type]))
    # y = np.asarray(y)
    y = np.asarray(sample['overall_sentiment_int'])

    # for i in range(len(image_name)):
    #     ydata[image_name[i]] = y[i]
    # return ydata,y.T
    return y.T

'''
def readTxtEmb(filename):
    path = "../data/"
    PATTERN = ' ,|,'
    f = open(path+filename)
    for line in f:
        line = line[:-1]
        row = re.split(PATTERN,line)
        imgname = row[0]
        emb = row[-768:]
        emb[0],emb[-1] = emb[0][1:], emb[-1][:-1]
        emb = [float(x) for x in emb]
        self.txt_emb_dict[imgname] = emb
'''

if __name__ == "__main__":
    dataset = readData('train',8000)
    sample = dataset[0][0]
    y = sampley2data(sample)
    unique, counts = np.unique(y, return_counts=True)
    _count_dict = dict(zip(unique, counts))
    print(_count_dict)
    x=list(_count_dict.values())/sum(_count_dict.values())
    perc = dict(zip(unique, list(x)))
    print(perc)

    # trial
    # {0: 21, 1: 59, 2: 302, 3: 445, 4: 173}
    # 5 classes: {0: 2.1%, 1: 5.9% 2: 30.2%, 3: 44.5%, 4: 17.3%}
    # 3 classes: {0: 8.0%, 1: 30.2%, 2: 61.8%}
    
    # train
    # {0: 151, 1: 479, 2: 2201, 3: 3127, 4: 1033}
    # 5 classes: {0: 2.16%, 1: 6.852%, 2: 31.48%, 3: 44.73%, 4: 14.78%}
    # 3 classes: {0: 9.01%, 1: 31.48%, 2: 59.51%}
