from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sklearn.linear_model
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.multiclass
from sklearn.datasets import make_multilabel_classification

import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score, f1_score

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample
from myDataLoader import MyDataLoader
import torch

from sklearn.metrics import f1_score


class LogisticRegression_MultiClass:
    def __init__(self):
        # # Assume X dim: (num_examples, num_features)
        # self.X_input = X
        # # Assume y dim: (num_examples, )
        # self.y_output = y

        self.model = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', warm_start=True)

    def fit(self, X_input, y_output):
        self.model.fit(X_input, y_output)

    def predict(self, input_X):
        return self.model.predict(input_X)

    def cal_accuracy(self, input_X, output_y):
        # Mean accuracy
        return self.model.score(input_X, output_y)



class LogisticRegression_MultiLabel:
    def __init__(self):
        # # Assume X dim: (num_examples, num_features)
        # self.X_input = X
        # # Assume y dim: (num_examples, num_total_classes), for each y, one-hot encoding, 1 if this sample is in this class, 0 if not
        # self.y_output = y

        self.model = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression(solver='lbfgs', warm_start=True))

    def fit(self, X_input, y_output):
        self.model.fit(X_input, y_output)

    def predict(self, input_X):
        return self.model.predict(input_X)

    def cal_accuracy(self, input_X, output_y):
        # Mean accuracy
        return self.model.score(input_X, output_y)



def getData(samples, imgname_textEmbs, is_task1, task1_class):
    if is_task1:
        X = []
        task1_y = []

        for image_n_i in range(len(samples['image_name'])):
            # Concat image pixels and text embeddings
            if samples['image_name'][image_n_i] in imgname_textEmbs:
                X.append(list(torch.flatten(samples['image'][image_n_i]).numpy()) + list(imgname_textEmbs[samples['image_name'][image_n_i]]))

                task1_y.append(samples[task1_class+'_int'][image_n_i])

        X_arr = np.asarray(X)
        task1_y_arr = np.asarray(task1_y)

        return X_arr, task1_y_arr
    else:
        X = []
        task2_y = []

        for image_n_i in range(len(samples['image_name'])):
            # Concat image pixels and text embeddings
            if samples['image_name'][image_n_i] in imgname_textEmbs:
                X.append(list(torch.flatten(samples['image'][image_n_i]).numpy()) + list(imgname_textEmbs[samples['image_name'][image_n_i]]))

                sample_2y = []

                for key in ['humour_int', 'sarcasm_int', 'offensive_int', 'motivational_int']:
                    if samples[key][image_n_i] >= 1:
                        sample_2y.append(1)
                    else:
                        sample_2y.append(0)

                task2_y.append(sample_2y)

        X_arr = np.asarray(X)
        task2_y_arr = np.asarray(task2_y)

        return X_arr, task2_y_arr

def eval_classifier(meme_dataset_transformed, imgname_textEmbs):
    batchSize = 700

    dataloader = DataLoader(dataset=meme_dataset_transformed, batch_size=batchSize,
    shuffle=True, num_workers=0)

    data_num = len(meme_dataset_transformed)

    training_batch_num = int(0.9*len(meme_dataset_transformed)) // batchSize

    # task1_classes = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
    task1_classes = ['overall_sentiment_ternary']

    for task1_class in task1_classes:
        print('Task 1 class: ', task1_class)

        train_X_arr = None
        train_task1_y_arr = None

        test_X_arr = []
        test_task1_y_arr = []

        lr_multiclass_classifier = LogisticRegression_MultiClass()

        for i_batch, sample in enumerate(dataloader):
            if i_batch < training_batch_num:
                # Partial fit
                X_arr, task1_y_arr = getData(sample, imgname_textEmbs, True, task1_class)

                lr_multiclass_classifier.fit(X_arr, task1_y_arr)

                train_X_arr = X_arr
                train_task1_y_arr = task1_y_arr
            else:
                X_arr, task1_y_arr = getData(sample, imgname_textEmbs, True, task1_class)

                test_X_arr += list(X_arr)
                test_task1_y_arr += list(task1_y_arr)

        test_X_arr = np.asarray(test_X_arr)
        test_task1_y_arr = np.asarray(test_task1_y_arr)

        task1_train_acc = lr_multiclass_classifier.cal_accuracy(train_X_arr, train_task1_y_arr)
        task1_test_acc = lr_multiclass_classifier.cal_accuracy(test_X_arr, test_task1_y_arr)

        task1_train_f1 = f1_score(train_task1_y_arr, lr_multiclass_classifier.predict(train_X_arr), average='macro')
        task1_test_f1 = f1_score(test_task1_y_arr, lr_multiclass_classifier.predict(test_X_arr), average='macro')

        print('Training accuracy: ', task1_train_acc)
        print('Testing accuracy: ', task1_test_acc)
        print('Testing f1: ', task1_test_f1)

    # print('Task 2: ')
    #
    # train_task2_y_arr = None
    # test_task2_y_arr = []
    #
    # lr_multilabel_classifier = LogisticRegression_MultiLabel()
    #
    # for i_batch, sample in enumerate(dataloader):
    #     if i_batch < training_batch_num:
    #         # Partial fit
    #         X_arr, task2_y_arr = getData(sample, imgname_textEmbs, False, '')
    #
    #         lr_multilabel_classifier.fit(X_arr, task2_y_arr)
    #
    #         train_X_arr = X_arr
    #         train_task2_y_arr = task2_y_arr
    #     else:
    #         X_arr, task2_y_arr = getData(sample, imgname_textEmbs, False, '')
    #
    #         test_X_arr += list(X_arr)
    #         test_task2_y_arr += list(task2_y_arr)
    #
    # test_X_arr = np.asarray(test_X_arr)
    # test_task2_y_arr = np.asarray(test_task2_y_arr)
    #
    # task2_train_acc = lr_multilabel_classifier.cal_accuracy(train_X_arr, train_task2_y_arr)
    # task2_test_acc = lr_multilabel_classifier.cal_accuracy(test_X_arr, test_task2_y_arr)
    #
    # print("Training accuracy", task2_train_acc)
    # print("Testing accuracy", task2_test_acc)

def get_transformed_dataset(textEmb_path, data_path, img_path):
    '''
    Get the embedding for the text, which is used as the text feature, and the dataset.
    '''

    imgname_textEmbs = MyDataLoader.read_text_embeddings_Idx(textEmb_path)

    data_transform = transforms.Compose([
      ResizeSample(size=(256, 256)),
      ToTensorSample(),
      NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    meme_dataset_transformed = MemeDataset(
      csv_file=os.path.join(os.getcwd(), data_path),
      image_dir = os.path.join(os.getcwd(), img_path),
        transform=data_transform)

    return imgname_textEmbs, meme_dataset_transformed

def main():
    trial_textEmb_path = '../data/data1_textEmbs.csv'
    trial_data_path = '../data/data1.csv'
    trial_img_path = '../data/semeval-2020_trialdata/Meme_images/'

    trial_imgname_textEmbs, trial_meme_dataset_transformed = get_transformed_dataset(trial_textEmb_path, trial_data_path, trial_img_path)

    print("Trial data: ")
    eval_classifier(trial_meme_dataset_transformed, trial_imgname_textEmbs)


    train_textEmb_path = '../data/data_7000_textEmbs.csv'
    train_data_path = '../data/data_7000_new.csv'
    train_img_path = '../data/memotion_analysis_training_data/data_7000/'

    train_imgname_textEmbs, train_meme_dataset_transformed = get_transformed_dataset(train_textEmb_path, train_data_path, train_img_path)

    print("Training data: ")
    eval_classifier(train_meme_dataset_transformed, train_imgname_textEmbs)



main()
