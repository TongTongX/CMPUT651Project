import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from models.infer_sent import InferSent
from models.model_utils import *


class CNNGloveAttentionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_classes = kwargs['num_classes']
        self.batch_size = kwargs['batch_size']

        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.img_batch_norm1 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 61 * 61, 1024)
        self.img_batch_norm2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.img_batch_norm3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 3)


        self.vocab_size = kwargs['vocab_size']
        self.embedding_dim = kwargs['embedding_dim']

        params_model = {'bsize': self.batch_size, 'word_emb_dim': self.embedding_dim, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(
          os.path.join(os.getcwd(), '../data/encoder/infersent1.pkl')))
        self.infersent.set_w2v_path(
          w2v_path=os.path.join(os.getcwd(), '../data/glove/glove.840B.300d.txt'))
        self.infersent.build_vocab_k_words(K=self.vocab_size)
        print('self.infersent:\n{}'.format(self.infersent))

        self.fc_size = 512
        self.encode_dim = 4096

        self.fc_text1 = nn.Linear(in_features=self.encode_dim,
          out_features=self.fc_size)

        self.batch_norm1 = nn.BatchNorm1d(self.fc_size)

        self.fc_text2 = nn.Linear(in_features=self.fc_size,
          out_features=512)

        self.batch_norm2 = nn.BatchNorm1d(512)

        self.fc_text3 = nn.Linear(in_features=512,
          out_features=self.num_classes)

        # Output layer for the concatenated features
        self.conc_fc = nn.Linear(in_features=self.num_classes+self.num_classes, out_features=self.num_classes)

        self.num_att_head = kwargs['att_head_num']

        self.W_att = nn.Linear(in_features=self.num_classes, out_features=self.num_att_head, bias=False)

    def forward(self, img_batch, text_batch):
        img_batch = F.relu(self.pool(self.conv1(img_batch)))
        img_batch = F.relu(self.pool(self.img_batch_norm1(self.conv2(img_batch))))

        img_batch = img_batch.view(-1, 16 * 61 * 61)

        img_batch = F.relu(self.img_batch_norm2(self.fc1(img_batch)))
        img_batch = F.relu(self.img_batch_norm3(self.fc2(img_batch)))

        # The predicted probabilistic of images
        img_pred_output = F.softmax(F.relu(self.fc3(img_batch)), dim=1)


        embeddings = self.infersent.encode(
          text_batch, bsize=self.batch_size, tokenize=False, verbose=False)
        embeddings = torch.FloatTensor(embeddings)

        embeddings = F.relu(self.batch_norm1(self.fc_text1(embeddings)))
        embeddings = F.relu(self.batch_norm2(self.fc_text2(embeddings)))

        # The predicted probabilistic of text
        text_pred_output = F.softmax(F.relu(self.fc_text3(embeddings)), dim=1)


        # Feature fusion, concatenate image and text features
        concat_features = torch.cat((img_pred_output, text_pred_output), dim=1)

        concat_pred_output = F.softmax(F.relu(self.conc_fc(concat_features)), dim=1)


        # Self attention to fuse the output desisions from image modal, text modal and concatenated modal
        img_pred_output_view = img_pred_output.view(img_pred_output.size()[0], 1, img_pred_output.size()[1])
        text_pred_output_view = text_pred_output.view(text_pred_output.size()[0], 1, text_pred_output.size()[1])
        concat_pred_output_view = concat_pred_output.view(concat_pred_output.size()[0], 1, concat_pred_output.size()[1])

        # combined_img_text_concat = torch.cat((img_pred_output_view, text_pred_output_view), dim=1)
        combined_img_text_concat = torch.cat((img_pred_output_view, text_pred_output_view, concat_pred_output_view), dim=1)

        att_img_text_concat =  F.softmax(F.relu(self.W_att(combined_img_text_concat)), dim=1)

        att_img_text_concat = att_img_text_concat.transpose(1, 2)

        weighted_img_text_concat_pred = att_img_text_concat@combined_img_text_concat

        avg_weighted_img_text_concat_pred = torch.sum(weighted_img_text_concat_pred, 1) / self.num_att_head

        avg_weighted_img_text_concat_pred = avg_weighted_img_text_concat_pred.view(len(text_batch), self.num_classes)

        return avg_weighted_img_text_concat_pred
