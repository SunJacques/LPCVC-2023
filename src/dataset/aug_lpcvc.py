import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms as T
import glob
import os
import numpy as np

class LPCVCDataset(Dataset):
    def __init__(self, datapath, augmentation=None, preprocessing=None,mean=0, std=1, n_class=14, train=True):
        self.datapath = datapath

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.n_class = n_class
        self.train = train

        self.mean = mean
        self.std = std
    
    def __len__(self):
        if self.train:
            files = glob.glob(os.path.join(self.datapath + 'train/IMG', "*.png"))
        else:
            files = glob.glob(os.path.join(self.datapath + 'val/IMG', "*.png"))
        return len(files)
    
    def __getitem__(self, idx):
        if self.train:
            img = cv2.imread(self.datapath + 'train/IMG/train_' + str(idx).zfill(4) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath  + 'train/GT/train_' +str(idx).zfill(4) + '.png')
        else:
            img = cv2.imread(self.datapath + 'val/IMG/val_' + str(idx).zfill(4) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath  + 'val/GT/val_' +str(idx).zfill(4) + '.png')
        
        mask = self.onehot(torch.as_tensor(np.array(mask), dtype=torch.int64), self.n_class)
        mask = np.transpose(mask, (1, 2, 0))

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']
        
            
        return img, mask
    
    def onehot(self, img, nb):
        oh = np.zeros((nb, img.shape[0], img.shape[1]))
        for i in range(nb):
            oh[i, :,:] = (img[:,:, 0] == i)
        return oh
