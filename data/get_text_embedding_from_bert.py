import numpy as np
import re
import csv
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def readCSV(filename):
    f = open(filename, 'r')
    dataset = []
    for line in f:
        line=line[:-1]
        row = re.split(' ,|,',line)
        dataset.append(row)

    return np.asarray(dataset)

def getTextEmb(text, tokenizer, model):
    marked_text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    sentence_embedding = torch.mean(encoded_layers[11], 1)

    return sentence_embedding


def getTextEmbeddings(dataset):

    f = open('data_output_multilabel_text_emb.csv', 'w')
    writer = csv.writer(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    for data_idx in range(dataset.shape[0]):
        print(data_idx)
        
        data = dataset[data_idx]

        text = None

        if len(data[3].strip()) == 0 and len(data[2].strip()) == 0:
            print("This data does not have OCR extracted text or actual text.")
            print(data)

        if len(data[3].strip()) != 0:
            # Use the actual text
            text = data[3].strip()
        elif len(data[3].strip()) == 0 and len(data[2].strip()) != 0:
            # Use the OCR extracted text
            text = data[2].strip()

        if text is not None:
            text_lower = text.lower()
            text_embedding = getTextEmb(text_lower, tokenizer, model).detach().cpu().numpy().flatten()

            text_embedding_str = ''
            for emb in text_embedding.tolist():
                text_embedding_str += str(emb) + ','

            data.append(text_embedding_str[:-1])

            writer.writerow(data)

dataset = readCSV("data_output_multilabel.csv")
getTextEmbeddings(dataset)
