import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):  #in_c = how many feature channel the input has/ out_c = how many do we want
        super().__init__()

        ### Layers definition


        # Convolution layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1) #change kernel_size if we want to use a 2*1 or other

        # Best normalization
        self.bn1 = nn.BatchNorm2d(out_c)


        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        # ReLu
        self.relu = nn.ReLU()

    # First Layer

    # Convolution Block
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

#Encoder Block
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        # Max Pooling
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p 

#Decoder Block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # Up convolution
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0) #cf paper for kernel_size, stride: we want to doublethe height
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip): #skip connection
        x = self.up(inputs)
        # Concatenation (channels)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder Blocks

        self.e1 = encoder_block(3,64)
        self.e2 = encoder_block(64,128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Botleneck Block

        self.b = conv_block(512,1024)

        # Decoder Block

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Classifier

        self.outputs = nn.Conv2d(64, 14, kernel_size=1, padding=0)


    def forward(self, inputs):

        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1) # p1 is  the output of the first encoder
        s3, p3 = self.e3(p2) 
        s4, p4 = self.e4(p3) 

        b = self.b(p4)  

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs

### Test

if __name__ == "__main__":
    inputs = torch.randn((10, 3,512,512))
    model = UNET()
    y = model(inputs)
    print(y.shape)