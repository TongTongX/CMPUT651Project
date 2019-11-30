import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os

from models.model_utils import *

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


class InceptRobertaAttentionModel(nn.Module):
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
        # dim: 2048
        print('self.num_ftrs: {}'.format(self.num_ftrs))
        self.inception.fc = nn.Linear(self.num_ftrs, self.num_classes)
        # Return features before fc layer.
        self.inception.fc = nn.Identity()
        print('self.inception:\n{}'.format(self.inception))
        self.image_size = 299

        self.img_classification_head = nn.Linear(self.num_ftrs, self.num_classes)


        # Text model
        self.vocab_size = kwargs['vocab_size']
        self.embedding_dim = kwargs['embedding_dim']  # 50


        self.roberta = RobertaModel.from_pretrained('../roberta.large', checkpoint_file='model.pt')
        self.roberta.eval()  # disable dropout (or leave in train mode to finetune)

        # set_parameter_requires_grad(self.roberta, True)

        self.roberta.register_classification_head('new_task', num_classes=self.num_classes)

        # Using the attention mechanism on this two embeddings and pass the weighted embedding to the fc
        self.num_att_head = kwargs['att_head_num']

        self.W_att = nn.Linear(in_features=self.num_classes, out_features=self.num_att_head, bias=False)

    def forward(self, image_batch, text_batch):
        # image_batch = sample_batch['image']
        if self.inception.training:
            image_features = self.inception(image_batch)[0]
        else:
            image_features = self.inception(image_batch)


        img_pred_output = F.softmax(F.relu(self.img_classification_head(image_features)), dim=1)


        token_batch = collate_tokens([self.roberta.encode(text_batch[text_idx]) for text_idx in range(len(text_batch))], pad_idx=1)

        text_logprobs = self.roberta.predict('new_task', token_batch)

        text_pred_output = torch.exp(text_logprobs)


        img_pred_output_view = img_pred_output.view(img_pred_output.size()[0], 1, img_pred_output.size()[1])
        text_pred_output_view = text_pred_output.view(text_pred_output.size()[0], 1, text_pred_output.size()[1])

        combined_img_text = torch.cat((img_pred_output_view, text_pred_output_view), dim=1)

        att_img_text =  F.softmax(self.W_att(combined_img_text), dim=1)

        att_img_text = att_img_text.transpose(1, 2)

        weighted_img_text_pred = att_img_text@combined_img_text

        avg_weighted_img_text_pred = torch.sum(weighted_img_text_pred, 1) / self.num_att_head

        avg_weighted_img_text_pred = avg_weighted_img_text_pred.view(len(text_batch), self.num_classes)

        return avg_weighted_img_text_pred
