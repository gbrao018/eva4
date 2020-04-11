import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, add_res_block = True, stride=1):
        super(ResBlock, self).__init__()
        self.add_res_block = add_res_block
        #ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] (Conv-BN-ReLU-Conv-BN-ReLU)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
            
    
    def forward(self, x):
        #ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] (Conv-BN-ReLU-Conv-BN-ReLU)
        if(self.add_res_block == False):
            return x
        else:
            R = F.relu(self.bn1(self.conv1(x))) 
            R = F.relu(self.bn2(self.conv2(R)))
            #X+R
            R += x
            return R
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv64 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn64 = nn.BatchNorm2d(64)
        
        #Layer1 -
        #X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        #R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] (Conv-BN-ReLU-Conv-BN-ReLU)
        #Add(X, R1)
        
        #input (32*32*3), output = (32*32*64), RF=2
        self.X64 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
            
        #input (32*32*64), output = (16*16*128), RF=5    
        self.X128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        #input (16*16*128), output = (8*8*256), RF=11    
        self.X256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
            
        #input (8*8*256), output = (4*4*512), RF=23    
        self.X512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
            
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(4,4) #input= (4*4*512), output(1*1*512) 
        
        self.layer1 = self._make_a11_layer(block,128, num_blocks[0],add_res_block = True) #input= (16*16*128), output(16*16*128)
        self.layer2 = self._make_a11_layer(block,256, num_blocks[1],add_res_block = False) #input= (8*8*256), output(8*8*256)
        self.layer3 = self._make_a11_layer(block,512, num_blocks[2],add_res_block = True) #input= (4*4*512), output(4*4*512)
        #self.layer3 = self._make_a11_layer(block,512, 512, num_blocks[3], X=self.X512)
        
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    
    def _make_a11_layer(self, block, out_planes,num_blocks, add_res_block, stride=1):
        
        R = block(out_planes,out_planes, add_res_block, stride)
        
        layers = []
        layers.append(R)
        return nn.Sequential(*layers)
        
    def forward(self, x):
        prepLayer = self.X64(x)
        X128 = self.X128(prepLayer)
        layer1 = self.layer1(X128)
        X256 = self.X256(layer1)
        layer2 = self.layer2(X256)
        X512 = self.X512(layer2)
        layer3 = self.layer3(X512)
        max_pool_4k = self.maxpool4(layer3)
        out = max_pool_4k.view(max_pool_4k.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
        
        
def ResNetA11():
    return ResNet(ResBlock, [2,2,2])
    