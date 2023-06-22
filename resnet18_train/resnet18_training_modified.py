#!/usr/bin/env python
# coding: utf-8

# # ResNet18 training
# 
# created on 20/06/2023

# In[1]:


# import dependencies
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
from torchvision import transforms as T
import glob
import os
import argparse
from tqdm import tqdm
import wandb
from torchvision import models
from torchsummary import summary


# ## Dataset Preparation

# In[12]:


class LPCVCDataset(Dataset):
    def __init__(self, datapath, transform, n_class=14, train=True, patch=False):
        self.datapath = datapath

        self.transform = transform
        self.n_class = n_class
        self.train = train

        self.patches = patch

    def __len__(self):
        if self.train:
            files = glob.glob(os.path.join(self.datapath + 'train/IMG/train', "*.png"))
        else:
            files = glob.glob(os.path.join(self.datapath + 'val/LPCVC_Val/IMG/val', "*.png"))
        return len(files)

    def __getitem__(self, idx):
        if self.train:
            img = cv2.imread(self.datapath + 'train/IMG/train/train_' + str(idx).zfill(4) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + 'train/GT_Updated/train/train_' + str(idx).zfill(4) + '.png')
        else:
            img = cv2.imread(self.datapath + 'val/LPCVC_Val/GT/val/val_' + str(idx).zfill(4) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + 'val/LPCVC_Val/IMG/val/val_' + str(idx).zfill(4) + '.png')

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        t = T.Compose([T.ToTensor(), T.Normalize(0, 1)])
        img = t(img)
        mask = self.onehot(mask, self.n_class)
        transpose_mask = np.transpose(mask, (2, 1, 0))

        # return img, mask
        return img, transpose_mask
    def onehot(self, img, nb):
        oh = np.zeros((img.shape[0], img.shape[1], nb))
        for i in range(nb):
            oh[:, :, i] = (img[:, :, 0] == i)
        return oh


# 

# ## Resnet Definition

# In[3]:


#define the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


"""
created on 20/06/2023, building a resnet18 by scratch
"""
class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        ## newly added parts
        self.upsample = nn.Upsample(scale_factor=32, mode = 'bilinear', align_corners=False)
        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)
        self.conv_out = nn.Conv2d(64 , num_classes, kernel_size=1, stride=1)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.conv_out(x)
        # x = self.avgpool(x)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


# In[5]:


resnet_model = ResNet_18(3, 14)


# In[7]:


summary(resnet_model,(3, 512, 512))


# ## Training

# In[8]:


#move the model to the device
resnet_model.to(device)
# next(resnet_model.parameters()).is_cuda


# ### Current testing block

# In[16]:


def train(model, args, train_loader):
    model.train()
    running_loss=0
    iteration=0
    correct = 0
    total=0

    for data in tqdm(train_loader):
        iteration+=1

        # inputs, labels = data[0].to(args.device), data[1].to(args.device)
        inputs, labels = data[0].to(args.device), data[1]

        with torch.cuda.amp.autocast():
            outputs=model(inputs)
            loss = args.criterion(outputs,labels)

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    train_loss=running_loss/len(train_loader)
    # train_loss=running_loss/31  # 1021/32
    accu=100.*correct/total


    train_accu.append(accu)
    train_losses.append(train_loss)

    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
    return(accu, train_loss)

def test(model, args, val_loader):
    model.eval()
    running_loss=0
    iteration=0
    correct = 0
    total=0

    with torch.no_grad():
        for data in tqdm(val_loader):
            iteration+=1

            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            with torch.cuda.amp.autocast():
                outputs=model(inputs)
                loss = args.criterion(outputs,labels)

            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


    test_loss=running_loss/len(val_loader)
    # test_loss=running_loss/3  # 100/32
    accu=100.*correct/total


    eval_accu.append(accu)
    eval_losses.append(test_loss)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
    return(accu, test_loss)


# Training settings
parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=0.0001)
# parser.add_argument('--dev', default="cuda:0")
parser.add_argument('--dev', default="cpu")
parser.add_argument('--momentum-sgd', type=float, default=0.9, metavar='M',
                    help='Momentum')
parser.add_argument('--datapath', default='dataset/')
args = parser.parse_args("")

args.device = torch.device(args.dev)
if args.dev != "cpu":
    torch.cuda.set_device(args.device)

model = resnet_model.to(args.device)

train_dataset = LPCVCDataset(datapath=args.datapath,transform=None,  n_class=14, train=True)
# train_dataset = drone_dataset_train
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        num_workers=0,
        pin_memory=True
)

# train_loader = train_loader

val_dataset = LPCVCDataset(datapath=args.datapath, transform=None,  n_class=14, train=False)
# val_dataset = drone_dataset_test
val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=1,
        num_workers=0,
        pin_memory=True
)

# val_loader = val_loader

args.criterion = torch.nn.BCEWithLogitsLoss().to(args.device)
args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)

train_accu = []
train_losses = []

eval_accu = []
eval_losses = []

# wandb.init(project="LPCVC", entity='lpcvc')
wandb.init(project="LPCVC")
wandb.run.name = "resnet18_train"
wandb.config.epochs = args.epochs
wandb.config.batch_size = args.batch_size
wandb.config.learning_rate = args.lr
wandb.config.weight_decay = args.weight_decay
wandb.config.momentum = args.momentum_sgd
wandb.config.train_dataset = train_dataset
wandb.config.test_dataset = val_dataset
# wandb.config.train_targets = train_dataset.targets


for epoch in range(1, args.epochs+1):
    print('\nEpoch : %d'%epoch)
    train_acc, train_loss = train(model, args, train_loader)
    test_acc, test_loss = test(model, args, val_loader)
    wandb.log(
        {"train_acc": train_acc, "train_loss": train_loss,
        "test_acc": test_acc, "test_loss": test_loss})

wandb.finish()


# In[112]:





# In[ ]:




