import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from src.dataset.vanilla_lpcvc import LPCVCDataset
import torchvision
import wandb

from src.model.model import UNET
from sample_solution.evaluation.accuracy import AccuracyTracker
from matplotlib.colors import ListedColormap


from tqdm import tqdm
import random
import numpy as np
import PIL
import albumentations as A
import cv2
import segmentation_models_pytorch as smp

accuracyTrackerTrain: AccuracyTracker = AccuracyTracker(n_classes=14)
accuracyTrackerVal: AccuracyTracker = AccuracyTracker(n_classes=14)

colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'lime', 'brown', 'gray', 'olive', 'teal', 'navy']
cmap = ListedColormap(colors[:15])

IMG_WIDTH = 128
IMG_HEIGHT = 128

mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]

def train(model, args, train_loader):
    model.train()

    running_loss = 0
    running_accu = 0
    running_dice = 0

    iteration=0

    loop = tqdm(train_loader)
    
    for batch_idx, (inputs, labels) in enumerate(loop):
        iteration+=1
        
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        

        outputs=model(inputs)
        loss = args.criterion(outputs,labels)
        
        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()
        args.sched.step() 

        loop.set_postfix(loss=loss.item())
        
        running_loss += loss.item()

        outputs = outputs.cpu().data.max(1)[1].numpy()
        labels = labels.cpu().data.max(1)[1].numpy()
        outputs.astype(np.uint8)
        labels.astype(np.uint8)

        accuracyTrackerTrain.update(labels, outputs)
        #running_accu += accuracyTrackerTrain.get_scores()
        #running_dice += accuracyTrackerTrain.get_mean_dice()


    train_loss = running_loss/iteration
    #train_accuracy = running_accu/iteration
    #train_dice = running_dice/iteration
        
    print('Train Loss: %.3f'%(train_loss))
    return(train_loss)

def eval(model, args, val_loader):
    model.eval()

    running_loss=0
    # running_accu = 0
    # running_dice = 0
    running_time = 0

    saved_images = np.zeros((3, 128, 128, 3))

    iteration=0

    loop = tqdm(val_loader)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loop):
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
            # running_accu += accuracyTrackerVal.get_scores()
            # running_dice += accuracyTrackerVal.get_mean_dice()

            if(batch_idx == 0):
                
                saved_images[0] = np.transpose(inputs.cpu().numpy()[0], (1, 2, 0))
                label = labels[0].reshape(128, 128, 1)
                output = outputs[0].reshape(128, 128, 1)
                saved_images[1] = cmap(np.repeat(label[:, :, np.newaxis], 3, axis=2).reshape(128, 128, 3))[:,:,0,:3]
                saved_images[2] = cmap(np.repeat(output[:, :, np.newaxis], 3, axis=2).reshape(128, 128, 3))[:,:,0,:3]


    val_loss=running_loss/iteration
    # val_accuracy = running_accu/iteration
    # val_dice = running_dice/iteration
    val_time = running_time/iteration

    print('Eval Loss: %.3f'%(val_loss))
    return(val_loss, val_time, saved_images)


def main():
    torch.cuda.empty_cache()
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dev', default="cuda:0")
    parser.add_argument('--momentum-sgd', type=float, default=0.9, metavar='M',
                        help='Momentum')
    parser.add_argument('--datapath', default='LPCVCDataset')
    args = parser.parse_args()

    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)

    #model = UNET(in_channels=3, out_channels=14, features=[64, 128, 256, 512]).to(args.device)
    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=14, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(args.device)

    transform = A.Compose([A.Resize(width=128, height=128, interpolation=cv2.INTER_NEAREST)])

    aug = A.Compose([A.Resize(128, 128, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])

    train_dataset = LPCVCDataset(datapath=args.datapath,mean=mean ,std=std ,n_class=14,transform=aug, train=True)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
    )
    val_dataset = LPCVCDataset(datapath=args.datapath, n_class=14,mean=mean ,std=std, transform=transform , train=False)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
    ) 
    args.criterion = torch.nn.CrossEntropyLoss().to(args.device)
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.1)
    args.scaler = torch.cuda.amp.GradScaler()
    args.sched = torch.optim.lr_scheduler.OneCycleLR(args.optimizer, args.lr, epochs=args.epochs,
                                            steps_per_epoch=len(train_loader))
    

    wandb.init(project="LPCVC")
    wandb.run.name = "UNET 0"
    wandb.config.epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.learning_rate = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.train_dataset_length = len(train_dataset)
    wandb.config.val_dataset_length = len(val_dataset)
    wandb.config.optmizer = "SGD"
    wandb.config.momentum = args.momentum_sgd


    for epoch in range(1, args.epochs+1):
        accuracyTrackerTrain.reset()
        accuracyTrackerVal.reset()
        print('\nEpoch : %d'%epoch)
        train_loss = train(model, args, train_loader)
        val_loss, val_time, saved_images = eval(model, args, val_loader)

        input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]
        wandb.log(
            {"train_acc": accuracyTrackerTrain.get_scores(), "train_loss": train_loss, "train_dice": accuracyTrackerTrain.get_mean_dice(),
            "val_acc": accuracyTrackerVal.get_scores(), "val_loss": val_loss, "val_dice": accuracyTrackerVal.get_mean_dice(), "inf_time": val_time, "input_image" : wandb.Image(input_image), "target_image" : wandb.Image(target_image), "pred_image" : wandb.Image(pred_image)})

        if(epoch%100==0):
            torch.save(model.state_dict(), 'src/model/vanilla-lpcvc_unet_'+str(epoch)+'_'+str(args.batch_size)+'.pth')

        scheduler.step()

wandb.finish()

if __name__ == '__main__':
    main()