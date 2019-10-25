import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score, f1_score

from data.meme_dataset import MemeDataset
from data.meme_transforms import ResizeSample, ToTensorSample, NormalizeSample

class MyDataLoader:
    def __init__(self, datalabel):
        self.data_transform = transforms.Compose([
            ResizeSample(size=(256, 256)),
            ToTensorSample(),
            NormalizeSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if datalabel == 'trail':
            self.dataset = MemeDataset(
                csv_file=os.path.join(os.getcwd(), '../data/data1.csv'),
                image_dir = os.path.join(os.getcwd(), '../data/semeval-2020_trialdata/Meme_images/'),
                transform= self.data_transform)
        else:
            self.dataset = MemeDataset(
                csv_file=os.path.join(os.getcwd(), '../data/data_7000_new.csv'),
                image_dir = os.path.join(os.getcwd(), '../data/memotion_analysis_training_data/data_7000/'),
                transform=self.data_transform)

        



