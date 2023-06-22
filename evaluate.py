import torch
import argparse
from torchvision import transforms as T
from src.dataset.vanilla_lpcvc import LPCVCDataset

from src.model.model import UNET
from sample_solution.evaluation.accuracy import AccuracyTracker
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import albumentations as A
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 128
mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]
colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'lime', 'brown', 'gray', 'olive', 'teal', 'navy']
cmap = ListedColormap(colors[:15])


accuracyTrackerVal: AccuracyTracker = AccuracyTracker(n_classes=14)

def eval(model, args, val_loader):
    model.eval()

    running_loss=0
    running_time = 0
    iteration=0

    saved_images = np.zeros((3, IMG_SIZE, IMG_SIZE, 3))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
            iteration+=1
            
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            starter.record()
            outputs=model(inputs)
            ender.record()

            torch.cuda.synchronize()

            running_time += starter.elapsed_time(ender)

            loss = args.criterion(outputs,labels)
            
            running_loss += loss.item()

            outputs = outputs.cpu().data.max(1)[1].numpy()
            labels = labels.cpu().data.max(1)[1].numpy()

            outputs.astype(np.uint8)
            labels.astype(np.uint8)

            accuracyTrackerVal.update(labels, outputs)

            if(batch_idx == 0):
                
                saved_images[0] = np.transpose(inputs.cpu().numpy()[0], (1, 2, 0))
                label = labels[0].reshape(IMG_SIZE, IMG_SIZE, 1)
                output = outputs[0].reshape(IMG_SIZE, IMG_SIZE, 1)
                saved_images[1] = cmap(np.repeat(label[:, :, np.newaxis], 3, axis=2).reshape(IMG_SIZE, IMG_SIZE, 3))[:,:,0,:3]
                saved_images[2] = cmap(np.repeat(output[:, :, np.newaxis], 3, axis=2).reshape(IMG_SIZE, IMG_SIZE, 3))[:,:,0,:3]


    val_loss=running_loss/iteration
    val_time = running_time/iteration

    print('Eval Loss: %.3f'%(val_loss))
    return(val_loss, val_time, saved_images)

def main():
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--modelpath', default="")
    parser.add_argument('--datapath', default='LPCVCDataset')
    args = parser.parse_args()

    model = torch.load(args.modelpath)

    transform = A.Compose([A.Resize(width=IMG_SIZE, height=IMG_SIZE, interpolation=cv2.INTER_NEAREST)])
    args.datapath = "/home/infres/jsun-22/LPCVC-2023/dataset/"

    val_dataset = LPCVCDataset(datapath=args.datapath, n_class=14,mean=mean ,std=std, transform=transform , train=False)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    val_loss, val_time, saved_images = eval(model, args, val_loader)
    input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]

    print("inf_time: {}, mean_dice: {}, score: {}".format(val_time, accuracyTrackerVal.get_mean_dice, accuracyTrackerVal.get_mean_dice/val_time))
    Image.fromarray(input_image).save("input")
    Image.fromarray(target_image).save("target")
    Image.fromarray(pred_image).save("pred")


if __name__ == '__main__':
    main()