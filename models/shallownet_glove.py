import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

import os

from models.infer_sent import InferSent
from models.model_utils import *


class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # print('x.size(): {}'.format(x.size()))
        x = self.pool(F.relu(self.conv1(x)))
        # print('x.size(): {}'.format(x.size()))
        x = self.pool(F.relu(self.conv2(x)))
        # print('x.size(): {}'.format(x.size()))
        x = x.view(-1, 16 * 53 * 53)
        # print('x.size(): {}'.format(x.size()))
        x = F.relu(self.fc1(x))
        # print('x.size(): {}'.format(x.size()))
        # x = F.relu(self.fc2(x)) # TODO may need to remove relu
        # print(x.shape)
        # x = self.fc3(x)   # TODO taking features only
        # print(x.shape)
        # input()
        return x


class ShallownetGloveModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_classes = kwargs['num_classes']
        self.batch_size = kwargs['batch_size']  # 64
        self.image_model = ShallowNet()
        self.image_size = 224

        # Text model
        self.vocab_size = kwargs['vocab_size']
        self.embedding_dim = kwargs['embedding_dim']  # 50

        params_model = {
            'bsize': self.batch_size, 'word_emb_dim': self.embedding_dim,
            'enc_lstm_dim': 2048, 'pool_type': 'max',
            'dpout_model': 0.0, 'version': 1}
        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(
        os.path.join(os.getcwd(), '../data/encoder/infersent1.pkl')))
        self.infersent.set_w2v_path(
        w2v_path=os.path.join(os.getcwd(), '../data/glove/glove.840B.300d.txt'))
        self.infersent.build_vocab_k_words(K=self.vocab_size)
        print('self.infersent:\n{}'.format(self.infersent))

        # Fully connected layers
        self.shallownet_output_dim = 120
        self.fc_size = 120
        self.encode_dim = 4096
        self.fc1 = nn.Linear(in_features=self.encode_dim, out_features=self.fc_size)
        self.fc2 = nn.Linear(in_features=self.fc_size+self.shallownet_output_dim,
                out_features=168)
        self.fc3 = nn.Linear(in_features=168, out_features=self.num_classes)
        print('self.fc1:\n{}'.format(self.fc1))
        print('self.fc2:\n{}'.format(self.fc2))

    def forward(self, image_batch, text_batch):
        image_features = self.image_model(image_batch)
        embeddings = self.infersent.encode(
            text_batch, bsize=self.batch_size, tokenize=False, verbose=False)
        embeddings = torch.FloatTensor(embeddings)
        text_features = F.relu(self.fc1(embeddings))
        # print('image_features.size(): {}, text_features.size(): {}'.format(
        #     image_features.size(), text_features.size()))
        concat_features = torch.cat((image_features, text_features), dim=1)
        # print('concat_features.size(): {}'.format(concat_features.size()))
        x = F.relu(self.fc2(concat_features))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x