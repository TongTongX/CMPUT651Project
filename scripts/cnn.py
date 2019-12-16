# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms

import syluutils as utils

classes = ('negative', 'neutral', 'positive')
imgpath='../data/memotion_analysis_training_data/data_7000/' 
datapath='../data/data_7000_new.csv'
batchsize=4

imgpath='../data/semeval-2020_trialdata/Meme_images/'
datapath='../data/data1.csv'

train_loader,test_loader = utils.getTrainTestLoader(datapath,batchsize)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
trainloader = iter(train_loader)
# images, _,labels = utils.getNextBatch(trainloader,imgpath)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))

# 2. Define a Convolutional Neural Network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.img_batch_norm1 = nn.BatchNorm1d(120)
        self.img_batch_norm2 = nn.BatchNorm1d(84)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 61 * 61)
        # print(x.shape)
        x = F.relu(self.img_batch_norm1(self.fc1(x)))
        # print(x.shape)
        x = F.relu(self.img_batch_norm2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        # input()
        return x


net = Net()

# 3. Define a Loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs,_, labels = utils.getBatchData(data,imgpath)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:

PATH = './cnn_imgs.pth'
torch.save(net.state_dict(), PATH)

testloader = iter(test_loader)
images, _, labels = utils.getNextBatch(testloader,imgpath)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))


net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batchsize)))


correct = 0
total = 0
y_pred = list()
y_true=list()
with torch.no_grad():
    for data in test_loader:
        images, _, labels = utils.getBatchData(data,imgpath)
        y_true+=list(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred+=list(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 700 test images: %.3f %%' % (
    100 * correct / total))

# y_pred = list(np.zeros(len(y_true)))
# y_pred=[x+2 for x in y_pred]
from sklearn.metrics import f1_score
# y_true = [int(x) for x in y_true]
# from collections import Counter
# print(Counter(y_true))
print('F1 score: %.3f %%' % (100*f1_score(y_true, y_pred, average = 'macro')))

class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))


with torch.no_grad():
    for data in test_loader:
        images, _, labels = utils.getBatchData(data,imgpath)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # print(c)
        for i in range(batchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print(class_total)

for i in range(3):
    print('Accuracy of %5s : %.3f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)