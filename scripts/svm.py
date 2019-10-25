from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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

# def readData(fileloc):
#     dataset = pd.read_csv(fileloc)
#     # to be edited: 
#     X = dataset.drop('Class', axis=1)
#     y = dataset['Class']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#     return X_train, X_test, y_train, y_test

class SVM_Classifier:
    def __init__(self, _kernel, _degree=None):
        # (kernal, degree) can be: ('linear'), ('poly', 6), ('rbf'), ('sigmoid')
        self.kernel = _kernel
        self.degree = _degree
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

        dataloader = DataLoader(dataset=dataset, batch_size=1,
            shuffle=True, num_workers=0)

        for i_batch, sample in enumerate(dataloader):
            print(i_batch)
            print(len(torch.flatten(sample['image'])))
            flat_img = torch.flatten(sample['image'])
            break

    def train(self, X_train, y_train):
        self.svclassifier.fit(X_train, y_train)
        return 

    def predictOnTest(self, X_test, y_test):
        y_pred = self.svclassifier.predict(X_test)
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        return
        
if __name__ == "__main__":
    svm = SVM_Classifier('linear')
    svm.readData('trial')
        
