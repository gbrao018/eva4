from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class QuizDNN(torch.nn.Module):
    def __init__(self, dropout_value = 0.1):
        """"Constructor of the class"""
        super(QuizDNN, self).__init__()
        # Input Block
        self.dropout_value = dropout_value
        #Depthwise separate 3 input channels into 3 output channels
        
        #self.conv1 = nn.Conv2d(3, 6, 5)  # RF= 
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        #torch.nn.Conv2d(in_channels, out_channels=k*in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
         
        self.layer1_depthwise = nn.Conv2d(in_channels=3, out_channels=1*3, kernel_size=3, stride=1, padding=2, dilation=2,  groups=3)
        # input=(32*32*3),kernel=((3*3*3),1,2,2),output=(32*32*1)*3 separate channels, Receptive Field=5
        
        self.layer1_pointwise = nn.Conv2d(1*3, 64, kernel_size=1,padding=0) 
        #input =(32*32*3), kernel=((1*1*1),stride=1,padding=0,dilation=1),output=(32*32*64), Receptive Field=5
        
        
        self.layer1_conv_block = nn.Sequential(
            self.layer1_depthwise,
            nn.ReLU(),
            self.layer1_pointwise,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) 
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = (16*16*64), RF=6
        self.layer64 = nn.Conv2d(64, 64, kernel_size=3,padding=1)
        self.layer128 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.layer256 = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        self.layer2_depthwise = nn.Conv2d(64, 1*64, kernel_size=3, stride=1, padding=1, dilation=1,  groups=64)
        # input=(16*16*64),kernel=((3*3*64),1,1,1),output=(16*16*1)*64 separate channels, Receptive Field=10
        
        self.layer2_pointwise = nn.Conv2d(1*64, 128, kernel_size=1,padding=0) 
        #input =(16*16*64), kernel=((1*1*1),stride=1,padding=0,dilation=1),output=(16*16*128), Receptive Field=10
        
        self.layer2_conv_block = nn.Sequential(
            self.layer2_depthwise,
            nn.ReLU(),
            self.layer2_pointwise,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) 
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = (8*8*128), RF=12

        self.layer3_depthwise = nn.Conv2d(128, 1*128, kernel_size=3, stride=1, padding=1, dilation=1,  groups=128)
        # input=(8*8*128),kernel=((3*3*128),1,1,1),output=(8*8*1)*128 separate channels, Receptive Field=20
        
        self.layer3_pointwise = nn.Conv2d(1*128, 256, kernel_size=1,padding=0) 
        #input =(8*8*128), kernel=((1*1*1),stride=1,padding=0,dilation=1),output=(8*8*256), Receptive Field=20
        self.layer4_depthwise = nn.Conv2d(256, 1*256, kernel_size=3, stride=1, padding=1, dilation=1,  groups=256)
        self.layer4_pointwise = nn.Conv2d(1*256, 512, kernel_size=1,padding=0)
        self.layer3_conv_block = nn.Sequential(
            self.layer3_depthwise,
            nn.ReLU(),
            self.layer3_pointwise,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.layer4_conv_block = nn.Sequential(
            self.layer4_depthwise,
            nn.ReLU(),
            self.layer4_pointwise,
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
            
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = (4*4*256), RF=24
        
        # OUTPUT BLOCK -> Add GAP Layer
        self.layer5_gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = (1*1*256), Receptive FieldF=48

        self.layer6_fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        #x = self.convblock1(x)
        x1 = x
        x2 = self.layer1_conv_block(x1)
        x1x2_cat = torch.cat((x1, x2), dim=1)
        x3 = self.self.layer64(x1x2_cat)
        x1x2x3_cat = torch.cat((x1x2_cat, x3), dim=1)
        x4 = self.pool1(x1x2x3_cat)
        x5 = self.layer2_conv_block(x4)
        x4x5_cat = torch.cat((x4, x5), dim=1)
        x6 = self.layer128(x4x5_cat)
        x4x5x6_cat = torch.cat((x4x5_cat, x6), dim=1)
        x7 = self.layer128(x4x5x6_cat)
        x5x6_cat = torch.cat((x5, x6), dim=1)
        x5x6x7_cat = torch.cat((x5x6_cat,x7), dim=1)
        x8 = self.pool2(x5x6x7_cat)
        x9 = self.layer3_conv_block(x8)
        x8x9_cat = torch.cat((x8,x9), dim=1)
        x10 = self.layer256(x8x9_cat)
        x8x9x10_cat = torch.cat((x8x9_cat,x10), dim=1)
        x11 = self.layer256(x8x9x10_cat)
        x12 = self.layer5_gap(x11)
        x13 = self.layer6_fc(x12)

        x = x13.view(-1, 10)
        return F.log_softmax(x13, dim=-1)
        