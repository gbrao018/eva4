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
        #it's first block uses following code:
        #x1 = Input
        #x2 = Conv(x1)
        #x3 = Conv(x1 + x2)
        #x4 = MaxPooling(x1 + x2 + x3)
        #x5 = Conv(x4)
        #x6 = Conv(x4 + x5)
        #x7 = Conv(x4 + x5 + x6)
        #x8 = MaxPooling(x5 + x6 + x7)
        #x9 = Conv(x8)
        #x10 = Conv (x8 + x9)
        #x11 = Conv (x8 + x9 + x10)
        #x12 = GAP(x11)
        #x13 = FC(x12)
        #Uses ReLU and BN wherever applicable
        #Uses CIFAR10 as the dataset
        #Your target is 75% in less than 40 Epochs
        #self.conv1 = nn.Conv2d(3, 6, 5)  # RF= 
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        #torch.nn.Conv2d(in_channels, out_channels=k*in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.Conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1) #input 32*32*3, output=32*32*64
        
        self.Conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #input 32*32*64, output=32*32*64
        
        self.Conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #input 16*16*64, output=16*16*64
        
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #input 8*8*64, output=8*8*64
        
        
        self.x2 = nn.Sequential( #input 32*32*64, output=32*32*64, RF=4
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        self.x3 = nn.Sequential(  #input 32*32*64, output=32*32*64, RF=6
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        
        self.x4 = nn.MaxPool2d(2, 2) #input 32*32*64, output=16*16*64, RF=7
        self.x5 = nn.Sequential( #input 16*16*64, output=16*16*64, RF=14
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        self.x6 = nn.Sequential( #input 16*16*64, output=16*16*64, RF=16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        self.x7 = nn.Sequential( #input 16*16*64, output=16*16*64, RF=18
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        
        self.x8 = nn.MaxPool2d(2, 2) #input 16*16*64, output_size = (8*8*64), RF=19
        self.x9 = nn.Sequential( #input 8*8*64, output_size = (8*8*64), RF=38
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        self.x10 = nn.Sequential( #input 8*8*64, output_size = (8*8*64), RF=40
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
        self.x11 = nn.Sequential( #input 8*8*64, output_size = (8*8*64), RF=42
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout(dropout_value)
        )
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

        self.dropout = nn.Dropout(self.dropout_value)

    
    def forward(self, x1):
        x2 = self.x2(x1)
        #print('x2=',len(x2))
        #x1_x2 = torch.cat((x1, x2), dim=1)
        x3 = self.x3(x2)
        #print('x3=',len(x3))
        #x1_x2_x3 = torch.cat((x2, x3),dim=2)
        x1_x2_x3 = x2.add(x3)
        x4 = self.x4(x1_x2_x3)
        #print('x4=',len(x4))
        x5 = self.x5(x4)
        #print('x5=',len(x5))
        #x4_x5 = torch.cat((x4, x5),dim=2)
        x4_x5 = x4.add(x5)
        x6 = self.x6(x4_x5)
        #print('x6=',len(x6))
        #x5_x6 = torch.cat((x5, x6),dim=2)
        x5_x6 = x5.add(x6)
        #x4_x5_x6 = torch.cat((x4, x5_x6),dim=2)
        x4_x5_x6 = x4.add(x5_x6)
        x7 = self.x7(x4_x5_x6)
        #print('x7=',len(x7))
        #x5_x6_x7 = torch.cat((x5_x6, x7),dim=2)
        x5_x6_x7 = x5_x6.add(x7)
        x8 = self.x8(x5_x6_x7)
        #print('x8=',len(x8))
        x9 = self.x9(x8)
        #print('x9=',len(x9))
        #x8_x9 = torch.cat((x8, x9),dim=2)
        x8_x9 = x8.add(x9)
        x10 = self.x10(x8_x9)
        #print('x10=',len(x10))
        #x8_x9_x10 = torch.cat((x8_x9, x10),dim=2)
        x8_x9_x10 = x8_x9.add(x10)
        x11 = self.x11(x8_x9_x10)
        #print('x11=',len(x11))
        x12 = self.x12_gap(x11)
        #print('x12=',len(x12))
        x12 = x12.view(-1, 64)
        x13 = self.x13_fc(x12)
        #print('x13=',len(x13))
        x13 = x13.view(-1, 10)
        #print('x13=',len(x13))
        #x14 = x13.view(-1, 10)
        return F.log_softmax(x13, dim=-1)