class ResNet(nn.Module):
    def __init__(self, ResBlock, n_classes, n_blocks_list=[3, 4, 6, 3],
                 out_channels_list=[64, 128, 256, 512], num_channels=3):
        """
        Args:
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_class: number of classes for image classifcation (used in classfication head)
            n_block_lists: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Create four convolutional layers
        in_channels = 64
        # For the first block of the second layer, do not downsample and use stride=1.
        self.conv2_x = self.CreateLayer(ResBlock, n_blocks_list[0], in_channels, out_channels_list[0], stride=1)
        
        # For the first blocks of conv3_x - conv5_x layers, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(ResBlock, n_blocks_list[1], out_channels_list[0]*ResBlock.expansion, out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(ResBlock, n_blocks_list[2], out_channels_list[1]*ResBlock.expansion, out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(ResBlock, n_blocks_list[3], out_channels_list[2]*ResBlock.expansion, out_channels_list[3], stride=2)

        # Average pooling (used in classification head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP for classification (used in classification head)
        self.fc = nn.Linear(out_channels_list[3] * ResBlock.expansion, n_classes)


    def forward(self, x):
        """
        Args: 
            x: input image
        Returns:
            C2: feature maps after conv2_x
            C3: feature maps after conv3_x
            C4: feature maps after conv4_x
            C5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        # Feature maps
        C2 = self.conv2_x(x)
        C3 = self.conv3_x(C2)
        C4 = self.conv4_x(C3)
        C5 = self.conv5_x(C4)

        # Classification head
        y = self.avgpool(C5)
        y = y.reshape(y.shape[0], -1)
        y = self.fc(y)

        return C2, C3, C4, C5, y


    def CreateLayer(self, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(ResBlock(in_channels, out_channels, stride=stride, is_first_block=True))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))

        return nn.Sequential(*layer)