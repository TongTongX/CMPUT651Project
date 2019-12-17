# CMPUT651Project
CMPUT 651 Project at University of Alberta in Fall 2019

## Setup Procudure
1. Ensure `torch` and `sklearn` are installed.
2. Set up the project.
```
$ pip install -e .
$ pip install torchtext
$ pip install spacy
$ python3 -m spacy download en
```
3. Download pre-trained [GloVe word vectors](https://nlp.stanford.edu/projects/glove/). Place the Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) word vectors in `/data/glove/` directory.
4. Download the [trial dataset](https://drive.google.com/drive/folders/1wLXyEM0q4l7X6mXjt6EwrN3RMG3Chpje?usp=sharing) and [training dataset](https://drive.google.com/folderview?id=10T60Od1lClzCss7CvZTwnwUEQf1CzWai), unzip the .zip files and move the image directories in `/data`.
5. Download the InferSent model trained with GloVe in `/data`.
```
$ mkdir encoder
$ curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```
6. Clone the Facebook AI Research Sequence-to-Sequence Toolkit (for PyTorch implementation of RoBERTa) in the project directory.
```
$ git clone git@github.com:pytorch/fairseq.git
```
7. Download the pretrained RoBERTa model `roberta.large` [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md). Decompress the file, and place the folder in the project directory.
