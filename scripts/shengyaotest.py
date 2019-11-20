import utils as utils

import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np

if __name__ == "__main__":
    # dataset, batch_num = utils.readData('trial',batch_size=1024)
    # # read all data
    # sample = dataset[0]
    # text = list(utils.sampletxt2data(sample).values())
    # label_name_dict = {'overall_sentiment_int':4}
    # y = list(utils.sampley2data(sample,label_name_dict).values())

    from torchtext.data import Field
    import re
    # tokenize = lambda x: re.split(' ,|,',x)
    tokenize = lambda x: re.split(' ',x)

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    from torchtext.data import TabularDataset

    tv_datafields = [("Image_name", LABEL), 
                    ("Image_URL", None), 
                    ("OCR_extracted_text",TEXT),
                    ("corrected_text", TEXT),
                    ("Humour", LABEL), 
                    ("Sarcasm", LABEL),
                    ("offensive", LABEL), 
                    ("Motivational", LABEL),
                    ("Overall_Sentiment", LABEL),
                    ("Basis_of_classification", None)]
    trn, vld = TabularDataset(
               path="../data/data1.csv", 
               format='csv',
               skip_header=True, 
               fields=tv_datafields).split(0.9)
    trn, test = trn.split(0.8)

    sentence = trn[0].corrected_text
    if sentence == ['','']:
        sentence = trn[0].OCR_extracted_text
    print(trn[0].Image_name,sentence[:-1])