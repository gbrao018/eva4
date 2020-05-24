from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class DenseDepthDNN2(torch.nn.Module):
    #This modal works with any image size.
    def __init__(self, dropout_value = 0.1):
        """"Constructor of the class"""
        super(DenseDepthDNN2, self).__init__()
        # Input Block
        self.dropout_value = dropout_value # This is to use if it overfits, but not yet used.
        self.x2 = nn.Sequential( #input 6*W*H, output=W*H*64, RF=2
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        #x3 = Conv(x1 + x2) , no x1
        self.x3 = nn.Sequential(  #input W*H*64, output=W*H*64, RF=4 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        
        #x4 maxpool (x2+x3) 
        self.x4 = nn.MaxPool2d(2, 2) #input W*H*128, output=W/2*H/2*128, RF=5
        
        #x5 = Conv(x4)
        self.x5 = nn.Sequential( #input W/2*H/2*128, output=W/2*H/2*128, RF=10
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            #nn.Dropout(dropout_value)
        )
        
        #x6 = Conv(x4 + x5)
        self.x6 = nn.Sequential( #input W/2*H/2*256, output=W/2*H/2*128, RF=12
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            #nn.Dropout(dropout_value)
        )

        #x7 = Conv(x4 + x5 + x6)
        self.x7 = nn.Sequential( #input W/2*H/2*384, output=W/2*H/2*128, RF=14
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            #nn.Dropout(dropout_value)
        )
        
        #x8 = MaxPooling(x5 + x6 + x7)
        self.x8 = nn.MaxPool2d(2, 2) #input W/2*H/2*384, output_size = (W/4*H/4*384), RF=15
        
        #x9 = Conv(x8)
        self.x9 = nn.Sequential( #input W/4*W/4*384, output_size = (W/4*H/4*256), RF=30
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
            #nn.Dropout(dropout_value)
        )
        #x10 = Conv (x8 + x9)
        self.x10 = nn.Sequential( #input W/4*H/4*640, output_size = (W/4*H/4*256), RF=32
            nn.Conv2d(in_channels=640, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
            #nn.Dropout(dropout_value)
        )
        
        #x11 = Conv (x8 + x9 + x10)
        self.x11 = nn.Sequential( #input W/4*H/4*896, output_size = (W/4*H/4*256), RF=34
            nn.Conv2d(in_channels=896, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
            #nn.Dropout(dropout_value)
        )

        #x12 = Conv (x9 + x10 + x11)
        self.x12 = nn.MaxPool2d(2, 2) #input W/4*H/4*768, output_size = (W/8*H/8*768), RF=35
        
        #x13 = Conv(x12)
        self.x13 = nn.Sequential( #input W/4*H/4*768, output_size = (W/4*H/4*512), RF=70
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            #nn.Dropout(dropout_value)
        )

        #x14 = Conv(x12 + x13)
        self.x14 = nn.Sequential( #input W/8*H/8*1280, output_size = (W/8*H/8*512), RF=72
            nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            #nn.Dropout(dropout_value)
        )

        #x15 = Conv(x12 + x13 + x14)
        self.x15 = nn.Sequential( #input W/8*H/8*1792, output_size = (W/8*H/8*512), RF=74
            nn.Conv2d(in_channels=1792, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            #nn.Dropout(dropout_value)
        )


        """
        # OUTPUT BLOCK -> Add GAP Layer
        self.x12_gap = nn.Sequential( # input 8*8*64, output_size = (1*1*64), Receptive FieldF=42
            nn.AvgPool2d(kernel_size=8)
        )

        self.x13_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=10)
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )
        """
        # We do not use Fully connected layers. Removed gap layer and added additional conv layers.
        
        self.xf1_invconv_512_256 = nn.Sequential( #input W/4*H/4*512, output_size = (W/4*H/4*256), RF=78
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )
        self.xf1_inconv_256 = nn.Sequential( #input W/4*W/4*256, output_size = (W/4*W/4*256), RF=80
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )
        
        self.xf2_invconv_256_128 = nn.Sequential( #input W/2*H/2*256, output_size = (W/2*H/2*128), RF=82
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )

        self.xf2_inconv_128 = nn.Sequential( #input W/2*H/2*128, output_size = (W/2*H/2*128), RF=84
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )
        
        self.xf3_invconv_128_64 = nn.Sequential( #input W*H*128, output_size = (W*H*64), RF=86
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )

        self.xf3_inconv_64 = nn.Sequential( #input W*H*64, output_size = (W*H*64), RF=88
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )

        self.x64_inconv_32 = nn.Sequential( #input 64*64*64, output_size = (64*64*32), RF=42
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        )

        #one for depth and one for mask
        self.xdepth_conv_1_1 = nn.Sequential( #input 64*64*32, output_size = (64*64*3), RF=42
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            #nn.Dropout(dropout_value)
        )
        self.predict_mask = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x1):
        
        
        
        
        x2 = self.x2(x1) #input 6*W*H, output=W*H*64 , RF =2                          
        x3 = self.x3(x2) #input W*H*64, output=W*H*64, RF=4  
        
        #Maxpool    
        x4 = self.x4(torch.cat([x2,x3],dim=1))#input W*H*128, output=W/2*H/2*128, RF=5
        x5 = self.x5(x4)   #input W/2*H/2*128, output=W/2*H/2*128, RF=10
        x6 = self.x6(torch.cat([x4,x5],dim=1))  #input W/2*H/2*256, output=W/2*H/2*128, RF=12
        
        x7 = self.x7(torch.cat([x4,x5,x6],dim=1)) #input W/2*H/2*384, output=W/2*H/2*128, RF=14
        x8 = self.x8(torch.cat([x5,x6,x7],dim=1)) #input W/2*H/2*384, output_size = (W/4*H/4*384), RF=15
        x9 = self.x9(x8) #input W/4*W/4*384, output_size = (W/4*H/4*256), RF=30
        x10 = self.x10(torch.cat([x8,x9],dim=1)) #input W/4*H/4*640, output_size = (W/4*H/4*256), RF=32
        x11 = self.x11(torch.cat([x8,x9,x10],dim=1)) #input W/4*H/4*896, output_size = (W/4*H/4*256), RF=34

        #Maxpool
        x12 = self.x12(torch.cat([x9,x10,x11],dim=1))#input W/4*H/4*768, output_size = (W/8*H/8*768), RF=35 
        
        x13 = self.x13(x12) #input W/4*H/4*768, output_size = (W/4*H/4*512), RF=70
        x14 = self.x14(torch.cat([x12,x13],dim=1)) #input W/8*H/8*1280, output_size = (W/8*H/8*512), RF=72
        x15 = self.x15(torch.cat([x12,x13,x14],dim=1)) #input W/8*H/8*1792, output_size = (W/8*H/8*512), RF=74
        
        # Now do upsampling
        up = Upsample(2).cuda() 
        xf1 = up.forward(x15) #input W/8*H/8*512, output_size = (W/4*H/4*512), RF=74
        xf1_invconv_512_256 = self.xf1_invconv_512_256(xf1) #input W/4*H/4*512, output_size = (W/4*H/4*256), RF=76
        xf1_inconv_256 = self.xf1_inconv_256(xf1_invconv_512_256.add(x11)) #input W/4*H/4*256, output_size = (W/4*H/4*256), RF=78
        
        xf2 = up.forward(xf1_inconv_256) #upscale #input W/4*H/4*256, output_size = (W/2*H/2*256), RF=78
        xf2_invconv_256_128 = self.xf2_invconv_256_128(xf2) #input W/2*H/2*256, output_size = (W/2*H/2*128), RF=80
        xf2_inconv_128 = self.xf2_inconv_128(xf2_invconv_256_128.add(x7)) #input W/2*H/2*128, output_size = (W/2*H/2*128), RF=82
        
        xf3 = up.forward(xf2_inconv_128) #upscale. #input W/2*H/2*128, output_size = (W*H*128), RF=82
        xf3_invconv_128_64 = self.xf3_invconv_128_64(xf3) #input W*H*128, output_size = (W*H*64), RF=84
        xf3_inconv_64 = self.xf3_inconv_64(xf3_invconv_128_64.add(x3)) #input W*H*64, output_size = (W*H*64), RF=86

        x64_inconv_32 = self.x64_inconv_32(xf3_inconv_64) #input W*H*64, output_size = (W*H*32), RF=88
        #{{finished upscaling}}
        
        #Use 1*1 kernel and create final output channel for depth. 
        xdepth_conv_1_1 = self.x32_conv_1_1(x64_inconv_32) #input W*H*32, output_size = (W*H*1), RF=88. Uses 1*1 kernel.
        predict_mask = torch.nn.functional.sigmoid(self.predict_mask(x64_inconv_32)) #input W*H*32, output_size = (W*H*1), RF=90. Uses 3*3 kernel.
        out_array = []
        out_array.append(xdepth_conv_1_1)
        out_array.append(predict_mask)
        return out_array
