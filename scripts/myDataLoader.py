import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score, f1_score

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

class MyDataLoader:
    def read_text_embeddings_Idx(filename):
        imgname_textEmbs = dict()

        f = open(filename, 'r')

        for row in f:
            row = row[:-1]

            row_lst = row.split(',')

            imagename = row_lst[0]
            textEmb_lst = row_lst[-768:]

            textEmb = np.zeros((768,))

            for emb_idx in range(len(textEmb_lst)):
                if emb_idx == 0:
                    textEmb[emb_idx] = float(textEmb_lst[emb_idx][1:])
                elif emb_idx == len(textEmb_lst) - 1:
                    textEmb[emb_idx] = float(textEmb_lst[emb_idx][:-1])
                else:
                    textEmb[emb_idx] = float(textEmb_lst[emb_idx])

            imgname_textEmbs[imagename] = textEmb

        return imgname_textEmbs
