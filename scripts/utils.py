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
    image_name = np.asarray(sample['image_name'])
    y = list() 
    for label_type in _LABEL_DICT:
        y.append(np.asarray(sample[label_type]))
    y = np.asarray(y).T

    for i in range(len(image_name)):
        ydata[image_name[i]] = y[i]
    return ydata

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

# if __name__ == "__main__":
#     dataset = readData('trial',3)
#     for x in getBatchSample(dataset,0):
#         print(x.values())
