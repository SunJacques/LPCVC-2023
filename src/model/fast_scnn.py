import torch
import torch.nn as nn

__all__=['FastSCNN']

class _DSConv(nn.Module):
    '''
    depthwise separable convolution
    '''
    def __init__(self,in_channels,out_channels,stride=1) -> None:
        super(_DSConv,self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,stride,padding=1,groups=in_channels),#depthwise, note the group param
            nn.BatchNorm2d(in_channels),
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
class _LearningToDownSample(nn.Module):
    '''
    learning to down sample module
    '''
    def __init__(self,inter_channels1,inter_channels2,out_channels):
        super(_LearningToDownSample,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,inter_channels1,kernel_size=3,stride=2),
            nn.BatchNorm2d(inter_channels1),
            nn.ReLU()
        )
        self.dsconv1 = _DSConv(inter_channels1,inter_channels2,stride = 2)
        self.dsconv2 = _DSConv(inter_channels2,out_channels,stride = 2)

    def forward(self,x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class _BottleneckBlock(nn.Module):
    '''
    bottleneck residual block
    '''
    def __init__(self,in_channels,out_channels,expansion_factor=6,stride=2):
        super(_BottleneckBlock,self).__init__()
        self.use_shortcut = (stride==1 and in_channels==out_channels)
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*expansion_factor,1),
            nn.BatchNorm2d(in_channels*expansion_factor),
            nn.ReLU()
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels*expansion_factor,in_channels*expansion_factor,3,
                    stride=stride,padding=1,groups=in_channels*expansion_factor),
            nn.BatchNorm2d(in_channels*expansion_factor),
            nn.ReLU()
        )
        self.linear_pointwise = nn.Sequential(
            nn.Conv2d(in_channels*expansion_factor,out_channels,1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self,x):
        out = self.pointwise_conv(x)
        out = self.depthwise_conv(out)
        out = self.linear_pointwise(out)
        if self.use_shortcut:
            out = x + out
        return out

class _ConvBNReLU(nn.Module):
    '''
    conv -> batchnorm -> relu
    '''
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0) -> None:
        super(_ConvBNReLU,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)


class _PoolingModule(nn.Module):
    def __init__(self,in_channels,out_channels,pool_size):
        super(_PoolingModule,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = _ConvBNReLU(in_channels,out_channels,1)
    
    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class _PPM(nn.Module):
    '''
    Pyramid pooling module
    '''
    def __init__(self,in_channels,out_channels) -> None:
        super(_PPM,self).__init__()
        inter_channels = int(in_channels/4)
        self.module1 = _PoolingModule(in_channels,inter_channels,1)
        self.module2 = _PoolingModule(in_channels,inter_channels,2)
        self.module3 = _PoolingModule(in_channels,inter_channels,3)
        self.module4 = _PoolingModule(in_channels,inter_channels,4)
        
        self.out = _ConvBNReLU(in_channels*2,out_channels,1)# in_channels + inter_channels*4
    
    def upsample(self,x,size):
        return torch.nn.functional.interpolate(x,size,mode='bilinear',align_corners=True)

    def forward(self,x):
        size = x.shape[2:]# x:(b,c,h,w)
        feature1 = self.upsample(self.module1(x),size)
        feature2 = self.upsample(self.module2(x),size)
        feature3 = self.upsample(self.module3(x),size)
        feature4 = self.upsample(self.module4(x),size)

        x = torch.cat([x,feature1,feature2,feature3,feature4],dim=1)
        return self.out(x)

class _GlobalFeatureExtractor(nn.Module):
    '''
    Global Feature Extractor
    '''
    def __init__(self,in_channels,inter_channels=(64,96,128),out_channels=128,t=6,block_nums=(3,3,3)):
        super().__init__()

        self.layer1 = self._BottleneckLayer(in_channels,inter_channels[0],block_nums[0],t,2)
        self.layer2 = self._BottleneckLayer(inter_channels[0],inter_channels[1],block_nums[1],t,2)
        self.layer3 = self._BottleneckLayer(inter_channels[1],inter_channels[2],block_nums[2],t,1)
        self.ppm = _PPM(inter_channels[2],out_channels)

    def _BottleneckLayer(self,in_channels,out_channels,block_num,t=6,stride=2):
        layers = []
        layers.append(_BottleneckBlock(in_channels,out_channels,t,stride))
        for i in range(1,block_num):
            layers.append(_BottleneckBlock(out_channels,out_channels,t,1))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.ppm(x)


class _FeatureFusionModule(nn.Module):
    '''
    Feature Fusion Module
    '''
    def __init__(self,higher_in_channels,lower_in_channels,out_channels,x=4):
        super(_FeatureFusionModule,self).__init__()
        self.x = x # scale factor
        self.dwconv = nn.Sequential(
            nn.Conv2d(lower_in_channels,out_channels,3,1,padding=x,dilation=x,groups=lower_in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(higher_in_channels,out_channels,1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self,higher_res_feature,lower_res_feature):
        lower = torch.nn.functional.interpolate(lower_res_feature,scale_factor=self.x, mode='bilinear',align_corners=True)
        lower = self.dwconv(lower)
        lower = self.conv_lower_res(lower)

        higher = self.conv_higher_res(higher_res_feature)

        return self.relu(lower+higher)

class _Classifier(nn.Module):
    '''
    final classifier
    '''
    def __init__(self,in_channels,num_classes,stride=1):
        super(_Classifier,self).__init__()
        self.dsconv1 = _DSConv(in_channels,in_channels,stride)
        self.dsconv2 = _DSConv(in_channels,in_channels,stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels,num_classes,1)
        )
    
    def forward(self,x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class FastSCNN(nn.Module):
    def __init__(self,nclass) -> None:
        '''
        param:
         nclass: the class num exclude the background
        '''
        super(FastSCNN,self).__init__()
        self.downsample = _LearningToDownSample(32,48,64)
        self.feature_extractor = _GlobalFeatureExtractor(in_channels=64,inter_channels=(64,96,128),out_channels=128,t=6,block_nums=(3,3,3))
        self.fusion = _FeatureFusionModule(higher_in_channels=64,lower_in_channels=128,out_channels=128)
        self.classifier = _Classifier(128,nclass+1)# add 1 to include the background

    def forward(self,x):
        size = x.shape[2:]
        higher_res = self.downsample(x)
        lower_res = self.feature_extractor(higher_res)
        x = self.fusion(higher_res,lower_res)
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(x,size,mode='bilinear',align_corners=True)
        return x

if __name__ == '__main__':
    model = FastSCNN(14)
    x = torch.randn(16,3,128,128)
    output = model(x)
    print(output.shape)