import torch
import torch.nn as nn
import torch.optim as optim
from models.shallownet_glove import ShallownetGloveModel
import syluutils as utils

classes = ('negative', 'neutral', 'positive')
imgpath='../data/memotion_analysis_training_data/data_7000/' 
datapath='../data/data_7000_new.csv'
batchsize=4

train_loader,test_loader = utils.getTrainTestLoader(datapath,batchsize)

deepsent_config = {
        'num_classes': 3, # negative, positive, neutral
        'batch_size': 4, 'vocab_size': 400000, 'embedding_dim': 300}
net = ShallownetGloveModel(**deepsent_config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)


for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, texts, labels = utils.getBatchData(data,imgpath)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
print('Finished Training')


testloader = iter(test_loader)
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, texts, labels = utils.getBatchData(data,imgpath)
        outputs = net(images, texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 20%% test images: %.3f %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for data in test_loader:
        images, texts, labels = utils.getBatchData(data,imgpath)
        outputs = net(images, texts)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # print(c)
        for i in range(batchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print(class_total)
for i in range(3):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
