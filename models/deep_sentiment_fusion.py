import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

from sklearn.decomposition import PCA

import sklearn

import os

from models.infer_sent import InferSent
from models.model_utils import *


class DeepSentimentFusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_classes = kwargs['num_classes']
        self.batch_size = kwargs['batch_size']
        # Pretrained image model
        self.inception = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model=self.inception, feature_extracting=True)
        # Handle the auxilary net
        self.num_ftrs = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(self.num_ftrs, self.num_classes)
        # Handle the primary net
        self.num_ftrs = self.inception.fc.in_features
        # dim: 2048
        print('self.num_ftrs: {}'.format(self.num_ftrs))
        self.inception.fc = nn.Linear(self.num_ftrs, self.num_classes)
        # Return features before fc layer.
        self.inception.fc = nn.Identity()
        # print('self.inception:\n{}'.format(self.inception))
        self.image_size = 299

        # Text model
        self.vocab_size = kwargs['vocab_size']
        self.embedding_dim = kwargs['embedding_dim']  # 50

        params_model = {'bsize': self.batch_size, 'word_emb_dim': self.embedding_dim, 'enc_lstm_dim': 2048,
                      'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(
        os.path.join(os.getcwd(), '../data/encoder/infersent1.pkl')))
        self.infersent.set_w2v_path(
        w2v_path=os.path.join(os.getcwd(), '../data/glove/glove.840B.300d.txt'))
        self.infersent.build_vocab_k_words(K=self.vocab_size)
        # print('self.infersent:\n{}'.format(self.infersent))

        self.encode_dim = 4096

        # Acc: 0.3

        # Reduce the feature len of img and text embs
        self.img_f_dim = 128
        self.text_emb_dim = 128

        self.fc_img = nn.Linear(self.num_ftrs, self.img_f_dim, bias=False)
        self.fc_text = nn.Linear(self.encode_dim, self.text_emb_dim, bias=False)

        self.fc1 = nn.Linear((self.img_f_dim+1)*(self.text_emb_dim+1), 128)
        # self.fc2 = nn.Linear(128, 128)
        self.out_f = nn.Linear(128, self.num_classes)

    def forward(self, image_batch, text_batch):
        # image_batch = sample_batch['image']
        if self.inception.training:
            image_features = self.inception(image_batch)[0]
        else:
            image_features = self.inception(image_batch)

        embeddings = self.infersent.encode(
        text_batch, bsize=self.batch_size, tokenize=False, verbose=True)
        embeddings = torch.FloatTensor(embeddings)

        image_features = self.fc_img(image_features)
        embeddings = self.fc_text(embeddings)

        # Tensor Fusion Layer: compute the outer product of img and text embs
        # https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
        img_f = torch.cat((Variable(torch.ones(self.batch_size, 1), requires_grad=False), image_features), dim=1)
        text_emb = torch.cat((Variable(torch.ones(self.batch_size, 1), requires_grad=False), embeddings), dim=1)
        # Dim: batch_size * 2049 * 4097
        fusion_tensor = torch.bmm(img_f.unsqueeze(2), text_emb.unsqueeze(1))

        # Flatten
        fusion_tensor = fusion_tensor.view(-1, (image_features.shape[1] + 1) * (embeddings.shape[1] + 1))

        x = F.relu(self.fc1(fusion_tensor))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.out_f(x))
        x = F.softmax(x, dim=1)

        return x
