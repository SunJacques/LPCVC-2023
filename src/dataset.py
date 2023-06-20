import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms as T
import glob
import os
import numpy as np

class LPCVCDataset(Dataset):
    def __init__(self, img_path, segm_path, transform,  n_class=14, train=True, patch=False):
        self.img_path = img_path
        self.segm_path = segm_path

        self.transform = transform
        self.n_class = n_class

        self.patches = patch
    
    def __len__(self):
        files = glob.glob(os.path.join(self.img_path, "*.png"))
        return len(files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + 'train_' + str(idx).zfill(4) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.segm_path  + 'train_' +str(idx).zfill(4) + '.png')

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        t = T.Compose([T.ToTensor(), T.Normalize(0, 1)])
        img = t(img)
        mask = self.onehot(mask, self.n_class)
            
        return img, mask
    
    def onehot(self, img, nb):
        oh = np.zeros((img.shape[0], img.shape[1], nb))
        for i in range(nb):
            oh[:,:,i] = (img[:,:,0] == i)
        return oh
