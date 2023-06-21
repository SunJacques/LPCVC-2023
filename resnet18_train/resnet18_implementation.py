"""
created on 20/06/2023, building a resnet18 by scratch
"""

import torch
import torch.nn as nn

class Block(nn.Module):
    def __int__(
            self, in_channels, intermediate_channels, identity_downsample=None, stride = 1
    ):
        super().__init__()
        self.expansion = 1 # there is no expansion in resnet18, but in other resnets
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1 # cuz the kernel size is 3
            )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1 # cuz the kernel size is 3
            )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.relu1 = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self,x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity # the skipping connection
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes): # image_channels: 3 for colored, 1 for grey
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64) # parameter: numbers of channels
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

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

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels): # used between two sub-layers between two layers
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
