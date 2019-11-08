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
            self.svclassifier = SVC(kernel=_kernel, degree=_degree, gamma='scale')
        else:
            self.svclassifier = SVC(kernel=_kernel, gamma='scale')
    
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
            emb = row[-768:]
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

    def splitData(self, dataset,batch_size):
        tr_idx = int(0.8*len(dataset)) // batch_size
        
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=False, num_workers=0)
        X_test, y_test = list(),list()
        start = False
        for i_batch, sample in enumerate(dataloader):
            X_batch, y_batch = self.sample2data(sample,batch_size) 
            if i_batch > tr_idx:
                if type(y_test) == list:
                    X_test = np.asarray(X_batch)
                    y_test = np.asarray(y_batch)
                else:
                    X_test = np.append(X_test,X_batch, axis=0)
                    y_test = np.append(y_test,y_batch, axis=0)
            else: 
                if start:
                    print(self.svclassifier.score(X_batch,y_batch))
                else:
                    start = True
                self.train(X_batch,y_batch)
        # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        # print(y_train.shape,y_test.shape)
        return X_test,y_test

    def train(self, X_train, y_train):
        self.svclassifier.fit(X_train, y_train)

    def test(self, X_test, y_test):
        # y_pred = self.svclassifier.predict(X_test)
        # print(confusion_matrix(y_test,y_pred))
        # print(classification_report(y_test,y_pred))
        print(self.svclassifier.score(X_test,y_test))
        
if __name__ == "__main__":
    label_name_dict = {'humour_int':0,'sarcasm_int':1,'offensive_int':2,'motivational_int':3,'overall_sentiment_int':4}
    # label_name_dict = {'motivational_int':3,'overall_sentiment_int':4}
    for key in label_name_dict.keys():
        print(key)
        svm = SVM_Classifier('linear',key,_degree=16)
        # svm.readTxtEmb("memotion_analysis_training_data/data_7000_textEmbs.csv")
        svm.readTxtEmb("semeval-2020_trialdata/data1_textEmbs.csv")
        # dataset = svm.readData('train')
        dataset = svm.readData('trial')
        X_test,y_test = svm.splitData(dataset,128)
        svm.test(X_test,y_test)
    
        
