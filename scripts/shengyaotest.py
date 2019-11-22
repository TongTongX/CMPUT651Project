import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(path):
    df = pd.read_csv(path, header=None)
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df = pd.DataFrame({
        'id':range(len(train_df)),
        'label':train_df[6],
        'alpha':['a']*train_df.shape[0],
        'text': train_df[1].replace(r'\n', ' ', regex=True)
    })
    dev_df = pd.DataFrame({
        'id':range(len(dev_df)),
        'label':dev_df[6],
        'alpha':['a']*dev_df.shape[0],
        'text': dev_df[1].replace(r'\n', ' ', regex=True)
    })

    train_df.to_csv('../data/train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
    dev_df.to_csv('../data/dev.tsv', sep='\t', index=False, header=False, columns=dev_df.columns)


if __name__ == "__main__":
    trial_path = '../data/merged_txt_trial.csv'
    train_path = '../data/merged_txt_train.csv'
    # preprocess(trial_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()


    
