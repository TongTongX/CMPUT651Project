import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os

from models.infer_sent import InferSent
from models.model_utils import *

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


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

        # params_model = {'bsize': self.batch_size, 'word_emb_dim': self.embedding_dim, 'enc_lstm_dim': 2048,
        #               'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # self.infersent = InferSent(params_model)
        # self.infersent.load_state_dict(torch.load(
        # os.path.join(os.getcwd(), '../data/encoder/infersent1.pkl')))
        # self.infersent.set_w2v_path(
        # w2v_path=os.path.join(os.getcwd(), '../data/glove/glove.840B.300d.txt'))
        # self.infersent.build_vocab_k_words(K=self.vocab_size)
        # # print('self.infersent:\n{}'.format(self.infersent))
        #
        # # set_parameter_requires_grad(self.infersent, True)

        self.encode_dim = 4096

        # # http://openaccess.thecvf.com/content_ICCV_2017/papers/Hori_Attention-Based_Multimodal_Fusion_ICCV_2017_paper.pdf
        # # Attention-based multimodal fusion
        # # Acc: 0.48 best from 20 epoch for trial, no fc, 128
        # # Acc:
        # self.c1 = nn.Linear(self.num_ftrs, 512, bias=False)
        # self.c2 = nn.Linear(self.encode_dim, 512, bias=False)
        #
        # self.att_Wc1 = nn.Linear(512, 256, bias=False)
        # self.att_Wc2 = nn.Linear(512, 256, bias=False)
        #
        # self.fc1 = nn.Linear(in_features=256, out_features=self.num_classes)



        self.roberta = RobertaModel.from_pretrained('../roberta.large', checkpoint_file='model.pt')
        self.roberta.eval()  # disable dropout (or leave in train mode to finetune)

        set_parameter_requires_grad(self.roberta, True)

        self.encode_dim = 1024
        self.hidden_dim = 256
        self.num_layers = 1

        self.rnn = nn.LSTM(input_size=self.encode_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=0)

        self.d_a = 300

        self.W_s1 = nn.Linear(in_features=2*self.hidden_dim, out_features=self.d_a, bias=False)
        self.W_s2 = nn.Linear(in_features=self.d_a, out_features=1, bias=False)

        self.fc1 = nn.Linear(self.num_ftrs+self.hidden_dim*2, 512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, image_batch, text_batch):
        # image_batch = sample_batch['image']
        if self.inception.training:
            image_features = self.inception(image_batch)[0]
        else:
            image_features = self.inception(image_batch)

        # embeddings = self.infersent.encode(
        # text_batch, bsize=self.batch_size, tokenize=False, verbose=False)
        # embeddings = torch.FloatTensor(embeddings)
        #
        # # Attention
        # c1_out = self.c1(image_features)
        # c2_out = self.c2(embeddings)
        #
        # Wc1_out = self.att_Wc1(c1_out)
        # Wc2_out = self.att_Wc2(c2_out)
        #
        # x = F.relu(Wc1_out + Wc2_out)
        # x = F.relu(self.fc1(x))
        # x = F.softmax(x, dim=1)



        token_batch = collate_tokens([self.roberta.encode(text_batch[text_idx]) for text_idx in range(len(text_batch))], pad_idx=1)

        # dim: batch_size, seq_len, emb_dim
        text_embeddings = self.roberta.extract_features(token_batch)

        # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068
        seq_len_lst = [text_embeddings.size()[1] for text_seq_idx in range(text_embeddings.size()[0])]

        lstm_input = nn.utils.rnn.pack_padded_sequence(text_embeddings, seq_len_lst, batch_first=True, enforce_sorted=False)

        lstm_outs, (hidden, cell) = self.rnn(lstm_input, None)
        #unpack sequence
        # output (Hidden) dim: batch_size, seq_len, num_directions*hidden_size
        output_Hidden, output_lengths = nn.utils.rnn.pad_packed_sequence(lstm_outs, batch_first=True)

        # https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py
        attention_output = torch.tanh(self.W_s1(output_Hidden))
        attention_output = self.W_s2(attention_output)
        attention_output = F.softmax(attention_output, dim=1)

        attention_output = attention_output.transpose(1, 2)

        sentence_emb = attention_output@output_Hidden
        sentence_emb = sentence_emb.view(len(text_batch), self.hidden_dim*2)

        concat_features = torch.cat((image_features, sentence_emb), dim=1)

        x = F.relu(self.fc1(concat_features))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)

        return x
