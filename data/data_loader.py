import numpy as np
import re

class DataLoader:
    def __init__(self):
        self.PATTERN = ' ,|,'

    def readCSV(self, filename):
        f = open(filename, 'r')
        dataset = []
        for line in f:
            line=line[:-1]
            row = re.split(self.PATTERN,line)
            dataset.append(row)

            # for testing
            # print("")
            # print("\n".join(row))
            # input()
            # # if len(row)<9:
            # #     print("")
            # #     print("\n".join(row))
            # #     input()
        return np.asarray(dataset)
        
    # data has been processed, this method won't be called any longer
    def preprocess_label(self,dataset):
        f = open('./data_output_multilabel.csv', 'w')

        VALID_LABELS = {
            'humour': ['not_funny', 'funny', 'very_funny', 'hilarious'],
            'sarcasm': ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted'],
            'offensive': ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive'],
            'motivational': ['not_motivational', 'motivational'],
            'overall_sentiment': ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']}

        for data in dataset:
            multilabel_onehot = [0, 0, 0, 0, 0]

            multilabel_onehot[0] = str(VALID_LABELS['humour'].index(data[4]))
            multilabel_onehot[1] = str(VALID_LABELS['sarcasm'].index(data[5]))
            multilabel_onehot[2] = str(VALID_LABELS['offensive'].index(data[6]))
            multilabel_onehot[3] = str(VALID_LABELS['motivational'].index(data[7]))
            multilabel_onehot[4] = str(VALID_LABELS['overall_sentiment'].index(data[8]))

            data[4:9] = multilabel_onehot
            f.write(",".join(data)+"\n")

if __name__ == "__main__":
    # readCSV("data_7000.csv")
    # dataset = readCSV("data_7000_new.csv")
    dataloader = DataLoader()
    dataset = dataloader.readCSV("data_output_multilabel.csv")
