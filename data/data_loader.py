import numpy as np 
import re
# data = """part 1;"this is ; part 2;";'this is ; part 3';part 4;this "is ; part" 5"""


def readCSV(filename):
    f = open(filename, 'r')
    dataset = []
    for line in f:
        line=line[:-1]
        row = re.split(' ,|,',line)
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

if __name__ == "__main__":
    dataset = readCSV("data_7000.csv")