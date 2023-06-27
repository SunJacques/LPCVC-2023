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
from PIL import Image

class LPCVCDataset(Dataset):
    def __init__(self, datapath, transform=None,mean=0, std=1, n_class=14, train=True):
        self.datapath = datapath

        self.transform = transform
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
            img = Image.open(self.datapath + 'train/IMG/train_' + str(idx).zfill(4) + '.png').convert('RGB')
            mask = Image.open(self.datapath + 'train/GT/train_' + str(idx).zfill(4) + '.png')
        else:
            img = Image.open(self.datapath + 'val/IMG/val_' + str(idx).zfill(4) + '.png').convert('RGB')
            mask = Image.open(self.datapath + 'val/GT/val_' + str(idx).zfill(4) + '.png')
        # img = np.array(img)
        # mask = np.array(mask)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # img = self.transform(img)
            # mask = self.transform(mask)
        
        t = T.Compose([T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = self.onehot(torch.as_tensor(np.array(mask), dtype=torch.int64), self.n_class)
            
        return img, mask
    
    def onehot(self, img, nb):
        oh = np.zeros((nb, img.shape[0], img.shape[1]))
        for i in range(nb):
            oh[i, :,:] = (img[:,:, 0] == i)
        return oh
