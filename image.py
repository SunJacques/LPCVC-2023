import torch
from src.dataset.vanilla_lpcvc import LPCVCDataset

import numpy as np
import cv2

import matplotlib.pyplot as plt
import albumentations as A

from sample_solution.evaluation.accuracy import AccuracyTracker
from matplotlib.colors import ListedColormap

accuracyTrackerVal: AccuracyTracker = AccuracyTracker(n_classes=14)

IMG_SIZE = 256
N = 4

mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]

colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'lime', 'brown', 'gray', 'olive', 'teal', 'navy']
cmap = ListedColormap(colors[:14])


model = torch.load('/home/infres/nvernier-22/project/LPCVC-2023/src/model/fpn/FPN+OC+256.pth')


aug_data = A.Compose([
        A.Resize(width=IMG_SIZE, height=IMG_SIZE, interpolation=cv2.INTER_NEAREST),
])


train_dataset = LPCVCDataset(datapath='/home/infres/nvernier-22/project/LPCVC-2023/dataset/',transform=aug_data, n_class=14, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for i in range(N):
    print('Segmenting image ' + str(i) + " / " + str(N))
    accuracyTrackerVal.reset()
    img, label = train_dataset[i]
    print(i)
    data = torch.tensor(np.expand_dims(img, axis=0)).to('cuda:1')
    outputs=model(data)


    labels = label.reshape(1, 14, IMG_SIZE, IMG_SIZE)
    labels = torch.tensor(labels).to('cuda:1')
    outputs = torch.tensor(outputs).to('cuda:1')
    labels = labels.reshape((1, 14,  IMG_SIZE, IMG_SIZE))

    labels = labels.cpu().data.max(1)[1].numpy()
    outputs = outputs.cpu().data.max(1)[1].numpy()
    labels.astype(np.uint8)
    outputs.astype(np.uint8)
    accuracyTrackerVal.update(labels, outputs)
    labels = np.squeeze(labels, axis=0)
    outputs = np.squeeze(outputs, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Affichage de la première image
    axes[0].imshow(np.transpose(img, (1, 2, 0)))
    axes[0].imshow(labels, vmin=0, vmax=13, cmap=cmap, alpha=0.4)
    axes[0].set_xlabel('Target')

    axes[1].imshow(np.transpose(img, (1, 2, 0)))
    axes[1].imshow(outputs, vmin=0, vmax=13, cmap=cmap, alpha=0.4)
    axes[1].set_xlabel('Prediction \n mean_dice = '+str(round(accuracyTrackerVal.get_mean_dice(), 3)) +'\n IoU = '+str(round(accuracyTrackerVal.get_scores(), 3)))

    axes[2].imshow(np.transpose(img, (1, 2, 0)))
    axes[2].set_xlabel('Input')
    plt.savefig('poster'+str(i)+'.png', dpi=300)
    #print(np.unique(labels))
