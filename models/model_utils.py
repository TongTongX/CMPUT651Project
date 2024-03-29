from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import time
import os
import copy
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
    is_inception=False, device=torch.device('cpu'), target_label='overall_sentiment_ternary_int'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0

    LOG_FILE = open('../LOG', 'w')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        LOG_FILE.write('Epoch ' + str(epoch) + '/' + str(num_epochs - 1) + '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_truth = []
            running_pred = []

            print('Phase: ', phase)

            batch_idx = 0

            # Iterate over data.
            for sample_batch in dataloaders[phase]:
                if batch_idx % 10 == 0:
                    print('Batch idx: ', batch_idx)
                batch_idx += 1

                image_batch = sample_batch['image'].to(device)
                corrected_text_batch = sample_batch['corrected_text']
                sentiment_labels = sample_batch[target_label].to(device)

                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    # TODO(xutong) may need to use aux_outputs
                    outputs = model(image_batch, corrected_text_batch)
                    loss = criterion(outputs, sentiment_labels)
                    '''
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    '''
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * image_batch.size(0)
                running_corrects += torch.sum(preds == sentiment_labels.data)
                running_truth += sentiment_labels.data.numpy().tolist()
                running_pred += preds.numpy().tolist()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(running_truth, running_pred)
            epoch_f1 = f1_score(running_truth, running_pred, average='macro')

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))

            LOG_FILE.write(phase + ' Loss: ' + str(epoch_loss) + ' Acc: ' + str(epoch_acc) + '\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val F1: {:4f}'.format(best_f1))

    LOG_FILE.write('Training complete in ' + str(time_elapsed // 60) + 'm ' + str(time_elapsed % 60) + 's' + '\n')
    LOG_FILE.write('Best val Acc: ' + str(best_acc) + '\n')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def train_att_model(model, dataloaders, criterion, optimizer, att_head_num, num_epochs=25,
    is_inception=False, device=torch.device('cpu'), target_label='overall_sentiment_ternary_int'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0

    LOG_FILE = open('../LOG_att_head_'+str(att_head_num), 'w')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        LOG_FILE.write('Epoch ' + str(epoch) + '/' + str(num_epochs - 1) + '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            f1_val = 0.0

            print('Phase: ', phase)

            batch_idx = 0

            # Iterate over data.
            for sample_batch in dataloaders[phase]:
                print('Batch idx: ', batch_idx)
                batch_idx += 1

                image_batch = sample_batch['image'].to(device)
                corrected_text_batch = sample_batch['corrected_text']
                sentiment_labels = sample_batch[target_label].to(device)

                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    # TODO(xutong) may need to use aux_outputs
                    outputs = model(image_batch, corrected_text_batch)
                    loss = criterion(outputs, sentiment_labels)
                    '''
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    '''
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * image_batch.size(0)
                running_corrects += torch.sum(preds == sentiment_labels.data)

                # F1 score
                f1_val += f1_score(sentiment_labels, preds, average='macro')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_f1_avg = f1_val / batch_idx

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1_avg))

            LOG_FILE.write(phase + ' Loss: ' + str(epoch_loss) + ' Acc: ' + str(epoch_acc) + ' F1: ' + str(epoch_f1_avg) + '\n')

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_f1_avg > best_f1:
                best_f1 = epoch_f1_avg
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_f1_avg)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Corresponding val Acc: {:4f}'.format(best_acc))

    LOG_FILE.write('Training complete in ' + str(time_elapsed // 60) + 'm ' + str(time_elapsed % 60) + 's' + '\n')
    LOG_FILE.write('Best val F1: ' + str(best_f1) + '\n')
    LOG_FILE.write('Corresponding val Acc: ' + str(best_acc) + '\n')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def train_balanced_att_model(model, dataloaders, imgpath, criterion, optimizer, att_head_num, num_epochs=25,
    is_inception=False, device=torch.device('cpu'), target_label='overall_sentiment_ternary_int'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # LOG_FILE = open('../LOG_att_head_'+str(att_head_num), 'w')
    # trainloader = iter(train_loader)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            images, texts, labels = utils.getBatchData(data,imgpath)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    correct = 0
    total = 0
    running_loss = 0.0

    f1_val = 0.0
    batch_idx = 0

    with torch.no_grad():
        for i, data in enumerate(dataloaders['val'], 0):
            batch_idx += 1

            images, texts, labels = utils.getBatchData(data,imgpath)
            outputs = model(images, texts)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # F1 score
            f1_val += f1_score(labels, predicted, average='macro')

            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('Accuracy of the network on the 20%% test images: %.3f %%' % (
        100 * correct / total))

    print('Macro F1 of the network on the 20%% test images: %.3f %%' % (
        100 * f1_val / batch_idx))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history



def train_svm_model(model, dataloaders, criterion, optimizer, num_epochs=25,
    is_inception=False, device=torch.device('cpu'), target_label='overall_sentiment_int'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    svm = SVC(kernel='rbf', gamma='scale')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_times = 0
            running_acc = 0.0

            # Iterate over data.
            for sample_batch in dataloaders[phase]:
                image_batch = sample_batch['image'].to(device)
                corrected_text_batch = sample_batch['corrected_text']
                sentiment_labels = sample_batch[target_label].to(device)

                concat_features = model(image_batch, corrected_text_batch)

                if phase == 'train':
                    svm.fit(concat_features, sentiment_labels)

                running_times += 1
                running_acc += svm.score(concat_features, sentiment_labels)

            epoch_acc = running_acc / running_times

            print('{} Avg Acc: {:.4f}'.format(phase, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def set_parameter_requires_grad(model, feature_extracting):
    """
    Args:
    feature_extracting: Flag for feature extracting. When False, we finetune the
    whole model, when True we only update the reshaped layer params
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, feature_only,
  use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    """ Inception v3
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    if feature_only:
        model_ft.fc = nn.Identity()
    input_size = 299

    return model_ft, input_size


def load_embedding_weights_glove(text_dir, emb_dir, filename):
    """Load the word embedding weights from a pre-trained model.

    Parameters:
        text_dir: The directory containing the text model.
        emb_dir: The subdirectory containing the weights.
        filename: The name of that text file.

    Returns:
        vocabulary: A list containing the words in the vocabulary.
        embedding: A numpy array of the weights.
    """
    vocabulary = []
    embedding = []
    with open(os.path.join(text_dir, emb_dir, filename), 'r') as f:
        for line in f.readlines():
            row = line.strip().split(' ')
            # Convert to unicode
            vocabulary.append(row[0])
            embedding.append([np.float32(x) for x in row[1:]])
        embedding = np.array(embedding)
        print('Finished loading word embedding weights.')
    return vocabulary, embedding
