# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynbjfruiequrueryewiuqryiwerqwe


import utils as utils

import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np

import pandas as pd

def reSaveTextData():
    dataset, batch_num = utils.readData('trial',batch_size=1024)
    # read all data
    sample = dataset[0]
    text_dict = utils.sampletxt2data(sample)
    text = list(text_dict.values())
    img_name = list(text_dict.keys())
    # label_name_dict = {'overall_sentiment_int':4}
    y_dict,y = utils.sampley2data(sample)
    df = pd.DataFrame(list(zip(img_name, text, list(y[0]),list(y[1]),list(y[2]),list(y[3]),list(y[4]))), 
               columns =['img_name','text', 'Humour','Sarcasm','offensive','Motivational','Overall_Sentiment']) 
    df.to_csv('merged_txt_trial.csv', sep=',', encoding='utf-8',index=False)

if __name__ == "__main__":
    # reSaveTextData()

    from torchtext.data import Field
    import re
    # tokenize = lambda x: re.split(' ,|,',x)
    tokenize = lambda x: re.split(' ',x)

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    from torchtext.data import TabularDataset

    tv_datafields = [('img_name', TEXT),
                    ("text",TEXT),
                    ("Humour", LABEL), 
                    ("Sarcasm", LABEL),
                    ("offensive", LABEL), 
                    ("Motivational", LABEL),
                    ("Overall_Sentiment", LABEL)]
    trn, vld = TabularDataset(
               path="merged_txt_trial.csv", 
               format='csv',
               skip_header=True, 
               fields=tv_datafields).split(0.9)
    trn, tst = trn.split(0.8)

    # sentence = trn[0].text
    # print(trn[0].img_name,sentence)

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(trn, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = "glove.6B.100d", 
                    unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(trn)

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (trn, vld, tst), 
        batch_size = BATCH_SIZE, 
        device = device)