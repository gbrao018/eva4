from __future__ import print_function
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
import Model
import torch.optim as optim

#g_transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

g_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
    
g_train_set = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=g_transform)

g_test_set = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=g_transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

class CIFARModel:
    def __init__(self, model, lr = 0.001, momentum = 0.9, step_size = 0, gamma = 0.01):
        self.m_train_losses = []
        self.m_test_losses = []
        self.m_train_acc = []
        self.m_test_acc = []
        self.m_model=copy.deepcopy(model)
        #self.m_optimizer=optim.SGD(self.m_model.parameters(), lr, momentum)
        #criterion = nn.CrossEntropyLoss()
        self.m_optimizer = optim.Adam(self.m_model.parameters(), lr)
        self.load_cifar_data(g_train_set,g_test_set)
        
    def clone_model(self):
        copy.deepcopy(self.m_model)
    
    def load_cifar_data(self,train_set,test_set):
        
        self.m_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                          shuffle=True, num_workers=2)
        self.m_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                         shuffle=False, num_workers=2)
                                         
