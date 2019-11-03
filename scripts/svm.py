from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

class SVM_Classifier:
    def __init__(self, _kernel, _ytype, _degree=None):
        # (kernal, degree) can be: ('linear'), ('poly', 6), ('rbf'), ('sigmoid')
        self.kernel = _kernel
        self.degree = _degree
        self.ytype = _ytype
        self.txt_emb_dict = dict()
        self.label_name_dict = {'humour_int':0,'sarcasm_int':1,'offensive_int':2,'motivational_int':3,'overall_sentiment_int':4}
        if _degree != None:
            self.svclassifier = SVC(kernel=_kernel, degree=_degree)
        else:
            self.svclassifier = SVC(kernel = _kernel)
    
    def readData(self, datalabel):
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
        return dataset

    def readTxtEmb(self,filename):
        path = "../data/"
        self.PATTERN = ' ,|,'
        f = open(path+filename)
        for line in f:
            line = line[:-1]
            row = re.split(self.PATTERN,line)
            imgname = row[0]
            emb = row[10:]
            emb[0],emb[-1] = emb[0][1:], emb[-1][:-1]
            emb = [float(x) for x in emb]
            self.txt_emb_dict[imgname] = emb

    def sample2data(self,sample, batch_size):

        flat_img_list = np.asarray([np.asarray(torch.flatten(x)) for x in sample['image']])
        image_name_list = np.asarray(sample['image_name'])
        txt_list = np.asarray([self.txt_emb_dict[x] for x in image_name_list])
        y = list()
        for label_type in self.label_name_dict.keys():
            y.append(np.asarray(sample[label_type]))

        y = np.asarray(y[self.label_name_dict[self.ytype]]).T
        X = np.append(flat_img_list, txt_list, axis=1)
        return X, y 

    def splitData(self, dataset, batch_size):
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=False, num_workers=0)
        X, y = list(),list()
         
        for i_batch, sample in enumerate(dataloader):
            X_batch, y_batch = self.sample2data(sample,batch_size) 
            if type(y) == list:
                X = np.asarray(X_batch)
                y = np.asarray(y_batch)
            else:
                X = np.append(X,X_batch, axis=0)
                y = np.append(y,y_batch, axis=0)
        X_train, y_train, X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        print(y_train.shape,y_test.shape)
        return X_train, y_train, X_test,y_test

    def train(self, X_train, y_train):
        self.svclassifier.fit(X_train, y_train)

    def test(self, X_test, y_test):
        y_pred = self.svclassifier.predict(X_test)
        print(confusion_matrix(y_batch,y_pred))
        print(classification_report(y_batch,y_pred))
        
if __name__ == "__main__":
    svm = SVM_Classifier('linear','humour_int')
    svm.readTxtEmb("semeval-2020_trialdata/data1_textEmbs.csv")
    dataset = svm.readData('trial')
    X_train, y_train, X_test,y_test = svm.splitData(dataset, 32)
    svm.train(X_train,y_train)
    svm.test(X_test,y_test)
    
        
