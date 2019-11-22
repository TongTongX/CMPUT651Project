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
        print('self.inception:\n{}'.format(self.inception))
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
        #
        # self.encode_dim = 4096



        self.roberta = RobertaModel.from_pretrained('../roberta.large', checkpoint_file='model.pt')
        self.roberta.eval()  # disable dropout (or leave in train mode to finetune)

        # set_parameter_requires_grad(self.roberta, True)

        # https://arxiv.org/pdf/1703.03130.pdf
        # A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
        self.encode_dim = 1024
        self.hidden_dim = 512
        self.num_layers = 1

        self.rnn = nn.LSTM(input_size=self.encode_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=0)

        self.d_a = 600
        self.att_head = 1

        self.W_s1 = nn.Linear(in_features=2*self.hidden_dim, out_features=self.d_a, bias=False)
        self.W_s2 = nn.Linear(in_features=self.d_a, out_features=self.att_head, bias=False)

        # Cast the image emb and the sentence emb in to the same dimension
        self.img_sen_dim = 512

        self.W_img = nn.Linear(in_features=self.num_ftrs, out_features=self.img_sen_dim, bias=False)
        self.W_sentence = nn.Linear(in_features=self.hidden_dim*2, out_features=self.img_sen_dim, bias=False)

        # Using the attention mechanism on this two embeddings and pass the weighted embedding to the fc
        self.W_att = nn.Linear(in_features=self.img_sen_dim, out_features=1, bias=False)

        # self.fc1 = nn.Linear(self.hidden_dim*2, 256)
        # self.fc1 = nn.Linear(self.num_ftrs + self.hidden_dim*2, 512)
        self.fc1 = nn.Linear(self.img_sen_dim, 512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

        # self.roberta.register_classification_head('new_task', num_classes=self.num_classes)

    def forward(self, image_batch, text_batch):
        # image_batch = sample_batch['image']
        if self.inception.training:
            image_features = self.inception(image_batch)[0]
        else:
            image_features = self.inception(image_batch)



        # embeddings = self.infersent.encode(
        # text_batch, bsize=self.batch_size, tokenize=False, verbose=False)
        # embeddings = torch.FloatTensor(embeddings)



        token_batch = collate_tokens([self.roberta.encode(text_batch[text_idx]) for text_idx in range(len(text_batch))], pad_idx=1)

        # dim: batch_size, seq_len, emb_dim
        text_embeddings = self.roberta.extract_features(token_batch)


        # # Just use the average emb of tokens as sentence emb
        # text_embeddings = torch.mean(text_embeddings, 1)



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
        sentence_emb = torch.sum(sentence_emb,1) / self.att_head

        # concat_features = torch.cat((image_features, sentence_emb), dim=1)


        img_cast_features = self.W_img(image_features)
        sent_cast_emb = self.W_sentence(sentence_emb)

        # Concat casted features into dim: batch_size, 2, self.img_sen_dim
        img_cast_features_view = img_cast_features.view(img_cast_features.size()[0], 1, img_cast_features.size()[1])
        sent_cast_emb_view = sent_cast_emb.view(sent_cast_emb.size()[0], 1, sent_cast_emb.size()[1])

        combined_img_sent = torch.cat((img_cast_features_view, sent_cast_emb_view), dim=1)

        att_img_sent =  F.softmax(self.W_att(torch.tanh(combined_img_sent)), dim=1)

        att_img_sent = att_img_sent.transpose(1, 2)

        weighted_img_sent = att_img_sent@combined_img_sent
        weighted_img_sent = weighted_img_sent.view(len(text_batch), self.img_sen_dim)


        # x = F.relu(self.fc1(sentence_emb))
        # x = F.relu(self.fc1(concat_features))
        x = F.relu(self.fc1(weighted_img_sent))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)

        # logprobs = self.roberta.predict('new_task', token_batch)
        # x = torch.exp(logprobs, dim=1)

        return x
