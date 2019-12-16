from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import re
import random

from collections import Counter
from imblearn.under_sampling import NeighbourhoodCleaningRule

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def readImg(path,imgname): 
    img = Image.open(path+imgname)
    img = img.convert('RGB')
    rsize = img.resize((256,256), Image.ANTIALIAS)
    # rsize.show()

    # define transformer
    transformer = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # to np array
    rsizeArr = np.asarray(rsize)
    # to tensor
    tensor_rsize = transformer(rsizeArr)
    # print(tensor_rsize)
    return tensor_rsize

def getAllData(path):
    PATTERN = ' ,|,'
    f = open(path)
    senti2label = {"positive":2, "very_positive":2, "very_negative":0, "negative":0, "neutral":1}
    dataset = list()

    for line in f:
        line = line[:-1]
        row = re.split(PATTERN,line)
        imgname = row[0]
        # empty correct text

        sample = [row[0], row[3],int(senti2label[row[-1]])]
        if row[3] == " ":
            sample = [row[0], row[2],int(senti2label[row[-1]])]
        dataset.append(sample)

    dataset = np.asarray(dataset)
    return dataset

def splitTrainTest(datapath):
    dataset = getAllData(datapath)
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(dataset, test_size=0.1)

    return trainset,testset

def getFullImgFeature(dataset):
    img_set = list()
    for sample in dataset:
        img_data = readImg(imgpath,sample[0])
        img_set.append(img_data)
    img_set = torch.stack(img_set,0)
    return img_set, dataset[:,-1]

def ncrReSample():
    raw_train, raw_test = splitTrainTest(datapath)
    img_data, y = getFullImgFeature(raw_train)
    print('Original dataset shape %s' % Counter(y))
    ncr = NeighbourhoodCleaningRule()
    X_res, y_res = ncr.fit_resample(img_data, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    trainset = np.append(X_res, y_res, axis=1)

    textX, texty = getFullImgFeature(raw_test)
    testset = np.append(textX, texty, axis=1)

    return trainset, testset

if __name__ == "__main__":
    imgpath='../data/memotion_analysis_training_data/data_7000/' 
    datapath='../data/data_7000_new.csv'
    batchsize=4

    raw_train, raw_test = splitTrainTest(datapath)
    img_data, y = getFullImgFeature(raw_train)
    print('Original dataset shape %s' % Counter(y))
    ncr = NeighbourhoodCleaningRule()
    X_res, y_res= ncr.fit_resample(img_data.reshape(-1, 1), y)
    print('Resampled dataset shape %s' % Counter(y_res))
    print(X_res)