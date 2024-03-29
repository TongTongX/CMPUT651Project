from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import re
import random

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
        # sample = [row[0], row[3],int(senti2label[row[-2]])]
        # if row[3] == " ":
        #     sample = [row[0], row[2],int(senti2label[row[-2]])]
        dataset.append(sample)

    dataset = np.asarray(dataset)
    return dataset

def getBalData(dataset):
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
    trainset,testset = splitTrainTest(datapath)

    trainset = getBalData(trainset)
    print(trainset.shape)
    print(testset.shape)

    trainloader = getDataLoader(trainset, batchsize)
    testloader = getDataLoader(testset, batchsize)

    return trainloader,testloader

def splitTrainTest(datapath):
    dataset = getAllData(datapath)
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(dataset, test_size=0.1)
    # print(trainset.shape)
    # print(testset.shape)

    # print(testset[:5])

    # trainloader = getDataLoader(trainset, batchsize)
    # testloader = getDataLoader(testset, batchsize)

    return trainset,testset

def overSamplingSMOTE(dataset):
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=27)
    X_train, y_train = sm.fit_sample(dataset[:,:-1], dataset[:,-1])
    print(y_train)


if __name__ == "__main__":
    imgpath='../data/memotion_analysis_training_data/data_7000/' 
    datapath='../data/data_7000_new.csv'
    batchsize=4

    # imgpath='../data/semeval-2020_trialdata/Meme_images/'
    # datapath='../data/data1.csv'

    # # USAGE: 
    # train_loader,test_loader = getTrainTestLoader(datapath,batchsize)
    # trainloader,testloader = iter(train_loader),iter(test_loader)
    # imgbatch, textbatch, y_batch = getNextBatch(trainloader,imgpath)
    # # OR
    # for i, data in enumerate(train_loader, 0):
    #     images, texts, labels = utils.getBatchData(data,imgpath)

    # dataset = getAllData(datapath)
    # loader = getDataLoader(dataset, batchsize)

    # correct = 0
    # total = 0
    # y_pred = list()
    # y_true=list()
    # with torch.no_grad():
    #     for data in loader:
    #         images, _, labels = getBatchData(data,imgpath)
    #         y_true+=list(labels)

    # y_pred = list(np.zeros(len(y_true)))
    # y_pred=[x+2 for x in y_pred]
    # from sklearn.metrics import f1_score
    # y_true = [int(x) for x in y_true]
    # from collections import Counter
    # print(Counter(y_true))
    # print('F1 score: %.3f %%' % (100*f1_score(y_true, y_pred, average = 'macro')))


    trainset,testset = splitTrainTest(datapath)
    trainset = getBalData(trainset)
    print(trainset.shape)
    print(testset.shape)
    # train_texts = trainset[:,1]
    # train_label = trainset[:,-1]
    # test_texts = testset[:,1]
    # test_label = testset[:,-1]

    import pandas as pd
    train_label_df = pd.DataFrame({'label':trainset[:,-1]})
    dev_label_df = pd.DataFrame({'label':testset[:,-1]})

    train_input0_df = pd.DataFrame({'text':trainset[:,1]})
    dev_input0_df = pd.DataFrame({'text':testset[:,1]})

    train_input0_df.to_csv('data/train.input0', index=False, header=False, columns=train_input0_df.columns)
    dev_input0_df.to_csv('data/dev.input0',  index=False, header=False, columns=dev_input0_df.columns)
    train_label_df.to_csv('data/train.label',  index=False, header=False, columns=train_label_df.columns)
    dev_label_df.to_csv('data/dev.label', index=False, header=False, columns=dev_label_df.columns)

