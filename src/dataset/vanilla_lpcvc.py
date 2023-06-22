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
    def __init__(self, datapath, transform=None,  n_class=14, train=True, patch=False):
        self.datapath = datapath

        self.transform = transform
        self.n_class = n_class
        self.train = train

        self.patches = patch
    
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
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        t = T.Compose([T.ToTensor(), T.Normalize(0, 1)])
        img = t(img)
        mask = self.onehot(torch.as_tensor(np.array(mask), dtype=torch.int64), self.n_class)
            
        return img, mask
    
    def onehot(self, img, nb):
        oh = np.zeros((nb, img.shape[0], img.shape[1]))
        for i in range(nb):
            oh[i, :,:] = (img[:,:, 0] == i)
        return oh
