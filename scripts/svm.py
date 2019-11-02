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

class SVM_Classifier:
    def __init__(self, _kernel, _degree=None):
        # (kernal, degree) can be: ('linear'), ('poly', 6), ('rbf'), ('sigmoid')
        self.kernel = _kernel
        self.degree = _degree
        self.label_name = ['humour_int','sarcasm_int','offensive_int','motivational_int','overall_sentiment_int']
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

    def sample2data(self,sample, batch_size):

        flat_img_list = np.asarray([np.asarray(torch.flatten(x)) for x in sample['image']])
        image_name_list = np.asarray(sample['image_name'])
        corrected_text = sample['corrected_text']
        ocr_extracted_text = sample['ocr_extracted_text']
        text_list = corrected_text.copy()
        for x in range(len(text_list)):
            if (text_list[x]) == 'nan':
                text_list[x] = ocr_extracted_text[x]
        text_list = np.asarray(text_list)
        y = list()
        for label_type in self.label_name:
            y.append(np.asarray(sample[label_type]))
        y=np.asarray(y).T

        return flat_img_list, text_list, y 

    def train(self, dataset, batch_size):

        tr_split_idx = int(0.8*len(dataset)) // batch_size
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=True, num_workers=0)
        
        for i_batch, sample in enumerate(dataloader):
            flat_img_batch, text_batch, y_batch = self.sample2data(sample,batch_size)

            # print(i_batch)
            # print(sample)
            # print(len(torch.flatten(sample['image'])))
            # print(flat_img_batch)
            print(y_batch)


            if i_batch>1:
                break

        # self.svclassifier.fit(X_train, y_train)
        return 

    def predictOnTest(self, X_test, y_test):
        y_pred = self.svclassifier.predict(X_test)
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        return
        
if __name__ == "__main__":
    svm = SVM_Classifier('linear')
    dataset = svm.readData('trial')
    svm.train(dataset,3)
    
        
