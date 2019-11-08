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



def getData(samples):
    X = []
    task1_y_humour = []
    task1_y_sarcasm = []
    task1_y_offensive = []
    task1_y_motivational = []
    task1_y_overall_sentiment = []
    task2_y = []

    for image_n_i in range(len(samples['image_name'])):
        # Concat image pixels and text embeddings
        if samples['image_name'][image_n_i] in imgname_textEmbs:
            X.append(list(torch.flatten(samples['image'][image_n_i]).numpy()) + list(imgname_textEmbs[samples['image_name'][image_n_i]]))

            task1_y_humour.append(samples['humour_int'][image_n_i])
            task1_y_sarcasm.append(samples['sarcasm_int'][image_n_i])
            task1_y_offensive.append(samples['offensive_int'][image_n_i])
            task1_y_motivational.append(samples['motivational_int'][image_n_i])
            task1_y_overall_sentiment.append(samples['overall_sentiment_int'][image_n_i])

            sample_2y = []

            for key in ['humour_int', 'sarcasm_int', 'offensive_int', 'motivational_int']:
                if samples[key][image_n_i] >= 1:
                    sample_2y.append(1)
                else:
                    sample_2y.append(0)

            task2_y.append(sample_2y)

    X_arr = np.asarray(X)
    task1_y_humour_arr = np.asarray(task1_y_humour)
    task1_y_sarcasm_arr = np.asarray(task1_y_sarcasm)
    task1_y_offensive_arr = np.asarray(task1_y_offensive)
    task1_y_motivational_arr = np.asarray(task1_y_motivational)
    task1_y_overall_sentiment_arr = np.asarray(task1_y_overall_sentiment)

    task2_y_arr = np.asarray(task2_y)

    return X_arr, task1_y_humour_arr, task1_y_sarcasm_arr, task1_y_offensive_arr, task1_y_motivational_arr, task1_y_overall_sentiment_arr, task2_y_arr

def eval_classifier(meme_dataset_transformed, imgname_textEmbs):
    batchSize = 700

    dataloader = DataLoader(dataset=meme_dataset_transformed, batch_size=batchSize,
    shuffle=True, num_workers=0)

    data_num = len(meme_dataset_transformed)

    training_batch_num = int(0.9*len(meme_dataset_transformed)) // batchSize

    train_X_arr = None
    train_task1_y_humour_arr = None
    train_task1_y_sarcasm_arr = None
    train_task1_y_offensive_arr = None
    train_task1_y_motivational_arr = None
    train_task1_y_overall_sentiment_arr = None
    train_task2_y_arr = None

    test_X_arr = []
    test_task1_y_humour_arr = []
    test_task1_y_sarcasm_arr = []
    test_task1_y_offensive_arr = []
    test_task1_y_motivational_arr = []
    test_task1_y_overall_sentiment_arr = []
    test_task2_y_arr = []

    lr_multiclass_humour_classifier = LogisticRegression_MultiClass()
    lr_multiclass_sarcasm_classifier = LogisticRegression_MultiClass()
    lr_multiclass_offensive_classifier = LogisticRegression_MultiClass()
    lr_multiclass_motivational_classifier = LogisticRegression_MultiClass()
    lr_multiclass_overall_sentiment_classifier = LogisticRegression_MultiClass()

    lr_multilabel_classifier = LogisticRegression_MultiLabel()

    for i_batch, sample in enumerate(dataloader):
        if i_batch < training_batch_num:
            # Partial fit
            X_arr, task1_y_humour_arr, task1_y_sarcasm_arr, task1_y_offensive_arr, task1_y_motivational_arr, task1_y_overall_sentiment_arr, task2_y_arr = getData(sample)

            lr_multiclass_humour_classifier.fit(X_arr, task1_y_humour_arr)
            lr_multiclass_sarcasm_classifier.fit(X_arr, task1_y_sarcasm_arr)
            lr_multiclass_offensive_classifier.fit(X_arr, task1_y_offensive_arr)
            lr_multiclass_motivational_classifier.fit(X_arr, task1_y_motivational_arr)
            lr_multiclass_overall_sentiment_classifier.fit(X_arr, task1_y_overall_sentiment_arr)

            lr_multilabel_classifier.fit(X_arr, task2_y_arr)

            train_X_arr = X_arr
            train_task1_y_humour_arr = task1_y_humour_arr
            train_task1_y_sarcasm_arr = task1_y_sarcasm_arr
            train_task1_y_offensive_arr = task1_y_offensive_arr
            train_task1_y_motivational_arr = task1_y_motivational_arr
            train_task1_y_overall_sentiment_arr = task1_y_overall_sentiment_arr
            train_task2_y_arr = task2_y_arr
        else:
            X_arr, task1_y_humour_arr, task1_y_sarcasm_arr, task1_y_offensive_arr, task1_y_motivational_arr, task1_y_overall_sentiment_arr, task2_y_arr = getData(sample)

            test_X_arr += list(X_arr)
            test_task1_y_humour_arr += list(task1_y_humour_arr)
            test_task1_y_sarcasm_arr += list(task1_y_sarcasm_arr)
            test_task1_y_offensive_arr += list(task1_y_offensive_arr)
            test_task1_y_motivational_arr += list(task1_y_motivational_arr)
            test_task1_y_overall_sentiment_arr += list(task1_y_overall_sentiment_arr)
            test_task2_y_arr += list(task2_y_arr)

    test_X_arr = np.asarray(test_X_arr)
    test_task1_y_humour_arr = np.asarray(test_task1_y_humour_arr)
    test_task1_y_sarcasm_arr = np.asarray(test_task1_y_sarcasm_arr)
    test_task1_y_offensive_arr = np.asarray(test_task1_y_offensive_arr)
    test_task1_y_motivational_arr = np.asarray(test_task1_y_motivational_arr)
    test_task1_y_overall_sentiment_arr = np.asarray(test_task1_y_overall_sentiment_arr)
    test_task2_y_arr = np.asarray(test_task2_y_arr)

    task1_humour_train_acc = lr_multiclass_humour_classifier.cal_accuracy(train_X_arr, train_task1_y_humour_arr)
    task1_humour_test_acc = lr_multiclass_humour_classifier.cal_accuracy(test_X_arr, test_task1_y_humour_arr)

    task1_sarcasm_train_acc = lr_multiclass_sarcasm_classifier.cal_accuracy(train_X_arr, train_task1_y_sarcasm_arr)
    task1_sarcasm_test_acc = lr_multiclass_sarcasm_classifier.cal_accuracy(test_X_arr, test_task1_y_sarcasm_arr)

    task1_offensive_train_acc = lr_multiclass_offensive_classifier.cal_accuracy(train_X_arr, train_task1_y_offensive_arr)
    task1_offensive_test_acc = lr_multiclass_offensive_classifier.cal_accuracy(test_X_arr, test_task1_y_offensive_arr)

    task1_motivational_train_acc = lr_multiclass_motivational_classifier.cal_accuracy(train_X_arr, train_task1_y_motivational_arr)
    task1_motivational_test_acc = lr_multiclass_motivational_classifier.cal_accuracy(test_X_arr, test_task1_y_motivational_arr)

    task1_overall_sentiment_train_acc = lr_multiclass_overall_sentiment_classifier.cal_accuracy(train_X_arr, train_task1_y_overall_sentiment_arr)
    task1_overall_sentiment_test_acc = lr_multiclass_overall_sentiment_classifier.cal_accuracy(test_X_arr, test_task1_y_overall_sentiment_arr)

    print("task1 humour training accuracy", task1_humour_train_acc)
    print("task1 humour testing accuracy", task1_humour_test_acc)

    print("task1 sarcasm training accuracy", task1_sarcasm_train_acc)
    print("task1 sarcasm testing accuracy", task1_sarcasm_test_acc)

    print("task1 offensive training accuracy", task1_offensive_train_acc)
    print("task1 offensive testing accuracy", task1_offensive_test_acc)

    print("task1 motivational training accuracy", task1_motivational_train_acc)
    print("task1 motivational testing accuracy", task1_motivational_test_acc)

    print("task1 overall_sentiment training accuracy", task1_overall_sentiment_train_acc)
    print("task1 overall_sentiment testing accuracy", task1_overall_sentiment_test_acc)

    task2_train_acc = lr_multilabel_classifier.cal_accuracy(train_X_arr, train_task2_y_arr)
    task2_test_acc = lr_multilabel_classifier.cal_accuracy(test_X_arr, test_task2_y_arr)

    print("task2 training accuracy", task2_train_acc)
    print("task2 testing accuracy", task2_test_acc)



imgname_textEmbs = MyDataLoader.read_text_embeddings_Idx('../data/data1_textEmbs.csv')

data_transform = transforms.Compose([
  ResizeSample(size=(256, 256)),
  ToTensorSample(),
  NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trial_meme_dataset_transformed = MemeDataset(
  csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
  image_dir = os.path.join(os.getcwd(), '../data/semeval-2020_trialdata/Meme_images/'),
    transform=data_transform)

print("trial_meme_dataset_transformed")
eval_classifier(trial_meme_dataset_transformed, imgname_textEmbs)


imgname_textEmbs = MyDataLoader.read_text_embeddings_Idx('../data/data_7000_textEmbs.csv')

train_meme_dataset_transformed = MemeDataset(
  csv_file=os.path.join(os.getcwd(), '../data/data_7000_new.csv'),
  image_dir = os.path.join(os.getcwd(), '../data/memotion_analysis_training_data/data_7000/'),
    transform=data_transform)

print("train_meme_dataset_transformed")
eval_classifier(train_meme_dataset_transformed, imgname_textEmbs)
