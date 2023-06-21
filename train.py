import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from src.dataset.vanilla_lpcvc import LPCVCDataset
import torchvision
import wandb

from src.model.unet import UNET

from tqdm import tqdm
import random
import numpy as np

def train(model, args, train_loader):
    model.train()
    running_loss=0
    iteration=0
    correct = 0
    total=0
    
    for data in tqdm(train_loader):
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


    train_loss=running_loss/len(train_loader)
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
    accu=100.*correct/total

    eval_accu.append(accu)
    eval_losses.append(test_loss)
        
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
    return(accu, test_loss)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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

    model = UNET().to(args.device)

    train_dataset = LPCVCDataset(datapath=args.datapath, n_class=14, train=True)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
    )
    val_dataset = LPCVCDataset(datapath=args.datapath, n_class=14, train=False)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
    ) 
    args.criterion = torch.nn.BCEWithLogitsLoss().to(args.device)
    args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)

    train_accu = []
    train_losses = []

    eval_accu = []
    eval_losses = []

    wandb.init(project="LPCVC")
    wandb.run.name = "UNET 0"
    wandb.config.epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.learning_rate = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.momentum = args.momentum_sgd
    wandb.config.train_dataset = train_dataset
    wandb.config.test_dataset = val_dataset
    wandb.config.train_targets = train_dataset.targets


    for epoch in range(1, args.epochs+1):
        print('\nEpoch : %d'%epoch)
        train_acc, train_loss = train(model, args, train_loader)
        test_acc, test_loss = test(model, args, val_loader)
        wandb.log(
            {"train_acc": train_acc, "train_loss": train_loss,
            "test_acc": test_acc, "test_loss": test_loss})

wandb.finish()


if __name__ == '__main__':
    main()