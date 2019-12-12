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

class DeepSentimentModel(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.num_classes = kwargs['num_classes']
    self.batch_size = kwargs['batch_size']  # 64
    # Pretrained image model
    self.inception = models.inception_v3(pretrained=True)
    # set_parameter_requires_grad(model=self.inception, feature_extracting=True)
    # Handle the auxilary net
    self.num_ftrs = self.inception.AuxLogits.fc.in_features
    self.inception.AuxLogits.fc = nn.Linear(self.num_ftrs, self.num_classes)
    # Handle the primary net
    self.num_ftrs = self.inception.fc.in_features
    print('self.num_ftrs: {}'.format(self.num_ftrs))
    self.inception.fc = nn.Linear(self.num_ftrs, self.num_classes)
    # Return features before fc layer.
    self.inception.fc = nn.Identity()
    # print('self.inception:\n{}'.format(self.inception))
    self.image_size = 299

    # Text model
    self.vocab_size = kwargs['vocab_size']
    self.embedding_dim = kwargs['embedding_dim']  # 50
    # self.glove_embedding = kwargs['glove_embedding']
    # # self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size,
    # #   embedding_dim=self.embedding_dim)
    # # self.word_embedding.load_state_dict({'weight': self.glove_embedding})
    # self.word_embedding = nn.Embedding.from_pretrained(
    #   embeddings=self.glove_embedding)

    params_model = {'bsize': self.batch_size, 'word_emb_dim': self.embedding_dim, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    self.infersent = InferSent(params_model)
    self.infersent.load_state_dict(torch.load(
      os.path.join(os.getcwd(), '../data/encoder/infersent1.pkl')))
    self.infersent.set_w2v_path(
      w2v_path=os.path.join(os.getcwd(), '../data/glove/glove.840B.300d.txt'))
    self.infersent.build_vocab_k_words(K=self.vocab_size)
    print('self.infersent:\n{}'.format(self.infersent))

    # LSTM
    self.hidden_size = 1024
    # self.lstm = nn.LSTM(input_size=4096,
    #   hidden_size=self.hidden_size, num_layers=1)
    # print('self.lstm:\n{}'.format(self.lstm))

    # Fully connected layers

    self.fc_size = 512
    self.encode_dim = 4096
    self.fc1 = nn.Linear(in_features=self.num_ftrs+self.encode_dim,
      out_features=self.fc_size)
    self.fc2 = nn.Linear(in_features=self.fc_size,
      out_features=self.num_classes)
    print('self.fc1:\n{}'.format(self.fc1))
    print('self.fc2:\n{}'.format(self.fc2))

  def forward(self, image_batch, text_batch):
    # image_batch = sample_batch['image']
    if self.inception.training:
      image_features = self.inception(image_batch)[0]
    else:
      image_features = self.inception(image_batch)  
    # image_features, _ = self.inception(image_batch)
    # print('image_features.size(): {}'.format(image_features.size()))
    
    # ocr_text_batch = sample_batch['ocr_extracted_text']
    # corrected_text_batch = sample_batch['corrected_text']
    # print('ocr_text_batch: {}\n'.format(ocr_text_batch))
    # print('corrected_text_batch: {}\n'.format(corrected_text_batch))
    # while 'nan' in corrected_text_batch:
    #   nan_idx = corrected_text_batch.index('nan')
    #   corrected_text_batch[nan_idx] = ocr_text_batch[nan_idx]
    # print('corrected_text_batch: {}\n'.format(corrected_text_batch))
    # numpy array with n vectors of dimension 4096
    embeddings = self.infersent.encode(
      text_batch, bsize=self.batch_size, tokenize=False, verbose=False)
    embeddings = torch.FloatTensor(embeddings)
    # print('embeddings.size(): {}'.format(embeddings.size()))
    # h_0 = c_0 = torch.zeros(1, self.batch_size, self.hidden_size)
    # print('h_0.size(): {}'.format(h_0.size()))
    # text_features, (h_n, c_n) = self.lstm(embeddings, (h_0, c_0))
    # print('text_features.size(): {}'.format(text_features.size()))

    # Concatenate image and text features
    concat_features = torch.cat((image_features, embeddings), dim=1)
    # print('concat_features.size(): {}'.format(concat_features.size()))
    x = F.relu(self.fc1(concat_features))
    # print('x.size(): {}'.format(x.size()))
    x = self.fc2(x)
    # print('x.size(): {}'.format(x.size()))
    x = F.softmax(x, dim=1)
    return x
  
  def initHidden(self):
    return torch.zeros(1, self.hidden_size)