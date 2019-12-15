import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from models.infer_sent import InferSent
from models.model_utils import *

import os

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


class CNNRobertaAttentionModel(nn.Module):
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


        # Text model
        self.vocab_size = kwargs['vocab_size']
        self.embedding_dim = kwargs['embedding_dim']  # 50


        self.roberta = RobertaModel.from_pretrained('../roberta.large', checkpoint_file='model.pt')
        self.roberta.eval()  # disable dropout (or leave in train mode to finetune)

        # set_parameter_requires_grad(self.roberta, True)

        self.roberta.register_classification_head('new_task', num_classes=self.num_classes)


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


        token_batch = collate_tokens([self.roberta.encode(text_batch[text_idx]) for text_idx in range(len(text_batch))], pad_idx=1)

        text_logprobs = self.roberta.predict('new_task', token_batch)

        # The predicted probabilistic of text
        text_pred_output = torch.exp(text_logprobs)


        # Feature fusion, concatenate image and text features
        concat_features = torch.cat((img_pred_output, text_pred_output), dim=1)

        concat_pred_output = F.softmax(F.relu(self.conc_fc(concat_features)), dim=1)


        # Self attention to fuse the output desisions from image modal, text modal and concatenated modal
        img_pred_output_view = img_pred_output.view(img_pred_output.size()[0], 1, img_pred_output.size()[1])
        text_pred_output_view = text_pred_output.view(text_pred_output.size()[0], 1, text_pred_output.size()[1])
        concat_pred_output_view = concat_pred_output.view(concat_pred_output.size()[0], 1, concat_pred_output.size()[1])

        combined_img_text_concat = torch.cat((img_pred_output_view, text_pred_output_view, concat_pred_output_view), dim=1)

        att_img_text_concat =  F.softmax(F.relu(self.W_att(combined_img_text_concat)), dim=1)

        att_img_text_concat = att_img_text_concat.transpose(1, 2)

        weighted_img_text_concat_pred = att_img_text_concat@combined_img_text_concat

        avg_weighted_img_text_concat_pred = torch.sum(weighted_img_text_concat_pred, 1) / self.num_att_head

        avg_weighted_img_text_concat_pred = avg_weighted_img_text_concat_pred.view(len(text_batch), self.num_classes)

        return avg_weighted_img_text_concat_pred
