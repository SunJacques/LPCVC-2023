import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from src.dataset.vanilla_lpcvc import LPCVCDataset
import torchvision

from src.model.unet import UNET

from tqdm import tqdm
import random
import numpy as np

def train(model, args, train_loader):
    model.train()
    running_loss=0
    iteration=0
    
    for data in tqdm(train_loader):
        iteration+=1
        
        inputs,labels=data[0].to(args.device),data[1].to(args.device)
        
        outputs=model(inputs)
        
        loss = args.criterion(outputs,labels)
        
        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()
        
        running_loss += loss.item()

    train_loss=running_loss/len(train_loader)
        
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss))
    return(train_loss)

def test(model, args, val_loader):
    model.eval()


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
    
    for epoch in range(1, args.epochs+1):
        print('\nEpoch : %d'%epoch)
        train(model, args, train_loader)
        if(epoch%20==0):
            #with torch.no_grad():
            #    test(model, args, val_loader)
            torch.save(model.state_dict(), 'models/vanilla-lpcvc_'+epoch+'.pth')
        if args.optimizer.param_groups[0]['lr'] < 0.001: break;



if __name__ == '__main__':
    main()