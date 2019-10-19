from data_loader import DataLoader
import numpy as np
from PIL import Image
import os
from matplotlib import cm

class ImgProcess:
    def __init__(self):
        pass
        
    def resizeImg(self, imgname):
        img = Image.open(imgname)
        img = img.convert('RGB')
        rsize = img.resize((256, 256), Image.ANTIALIAS)
        rsizeArr = np.asarray(rsize)
        return rsizeArr
    
    def normalize(self, rsizeArr):
        flat_arr = []
        for i in range(3):
            channel = rsizeArr[:,:,i]
            channel = channel.astype('float64')
            _min = np.amin(channel)
            _max = np.amax(channel)
            # normalize
            arr = 2*(channel-_min)/(_max-_min) - 1
            fc_arr = list(arr.flatten())
            flat_arr = flat_arr + fc_arr
        return np.asarray(flat_arr).astype('float16')
    
    def process(self, imgname):
        rsizeArr = self.resizeImg(imgname)
        arr = self.normalize(rsizeArr)
        return arr

if __name__ == "__main__":

    path = "../data/data_7000"
    files= os.listdir(path)

    f = open("img_norm.csv",'w')
    count = 0
    for file in files: 
        if not os.path.isdir(file): 
            imgname = path+"/"+file
            imgprocessor = ImgProcess()
            arr = imgprocessor.process(imgname)
            arr = arr.astype('str')
            f.write(file +","+ ",".join(arr)+"\n")
            print(count)
            count +=1