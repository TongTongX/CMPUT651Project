import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os

from models.infer_sent import InferSent
from models.model_utils import *


class DeepSentimentAttentionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_classes = kwargs['num_classes']
        self.batch_size = kwargs['batch_size']  # 64
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

        # http://openaccess.thecvf.com/content_ICCV_2017/papers/Hori_Attention-Based_Multimodal_Fusion_ICCV_2017_paper.pdf
        # Attention-based multimodal fusion
        # Acc: 0.425
        self.c1 = nn.Linear(self.num_ftrs, 128, bias=False)
        self.c2 = nn.Linear(self.encode_dim, 128, bias=False)

        self.att_Wc1 = nn.Linear(128, self.num_classes, bias=False)
        self.att_Wc2 = nn.Linear(128, self.num_classes, bias=False)

        # self.fc1 = nn.Linear(in_features=64, out_features=self.num_classes)

    def forward(self, image_batch, text_batch):
        # image_batch = sample_batch['image']
        if self.inception.training:
            image_features = self.inception(image_batch)[0]
        else:
            image_features = self.inception(image_batch)

        embeddings = self.infersent.encode(
        text_batch, bsize=self.batch_size, tokenize=False, verbose=True)
        embeddings = torch.FloatTensor(embeddings)

        # Attention
        c1_out = self.c1(image_features)
        c2_out = self.c2(embeddings)

        Wc1_out = self.att_Wc1(c1_out)
        Wc2_out = self.att_Wc2(c2_out)

        x = F.relu(Wc1_out + Wc2_out)
        # x = F.relu(self.fc1(x))
        x = F.softmax(x, dim=1)

        return x
