from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import re
import random

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

def getLeastLabel(dataset):
    iDataset = list(dataset[:,2])
    min_count = len(dataset)
    for i in range(3):
        count = iDataset.count(str(i))
        if count < min_count:
            min_count = count
    return min_count

def getEqualLabels(dataset, min_label):
    output = list()
    for i in range(3):
        output += (random.choices([sample for sample in dataset if sample[-1] == str(i)],k=min_label))
    output = np.asarray(output)
    np.random.shuffle(output)
    return output

def getBalData(path):
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
    min_label = getLeastLabel(dataset)
    bal_dataset = getEqualLabels(dataset,min_label) 
    # print(bal_dataset.shape)
    return bal_dataset

def getDataLoader(dataset, batchsize):
    dataset = np.resize(dataset, (dataset.shape[0]//batchsize,batchsize,3))
    return dataset

def getNextBatch(dataloader, imgpath):
    batch = next(dataloader)
    imgbatch, textbatch, y_batch = getBatchData(batch, imgpath)
    return [imgbatch, textbatch, y_batch]

def getBatchData(batch, imgpath):
    img_batch = list()
    for sample in batch:
        img_data = readImg(imgpath,sample[0])
        img_batch.append(img_data)
    imgbatch = torch.stack(img_batch,0)

    y_batch = torch.Tensor(batch[:,-1].astype("int")).to(dtype=torch.int64)
    # print(y_batch)
    return [imgbatch, batch[:,1], y_batch]

def getTrainTestLoader(datapath,batchsize):
    dataset = getBalData(datapath)
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(dataset, test_size=0.2)
    # print(trainset.shape)
    # print(testset.shape)

    trainloader = getDataLoader(trainset, batchsize)
    testloader = getDataLoader(testset, batchsize)

    return trainloader,testloader

if __name__ == "__main__":
    imgpath='../data/memotion_analysis_training_data/data_7000/' 
    datapath='../data/data_7000_new.csv'
    batchsize=8

    # USAGE: 
    trainloader,testloader = getTrainTestLoader(datapath,batchsize)
    trainloader,testloader = iter(trainloader),iter(testloader)
    imgbatch, textbatch, y_batch = getNextBatch(trainloader,imgpath)
    # OR
    for i, data in enumerate(trainloader, 0):
        images, texts, labels = utils.getBatchData(data,imgpath)

    # print(next(testloader))
    # print(next(testloader))

    # y = torch.Tensor(dataset[:,-1].astype(int))
    # readImg(imgpath,"10_year_13-2.jpg")