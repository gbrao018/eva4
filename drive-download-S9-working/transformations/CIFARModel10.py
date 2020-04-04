from __future__ import print_function
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
import torch.optim as optim

#g_transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#g_transform = transforms.Compose([
#    transforms.Pad(4),
#    transforms.RandomHorizontalFlip(),
#    transforms.RandomCrop(32),
#    transforms.ToTensor()])

g_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

g_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

g_train_set = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=g_transform_train)

g_test_set = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=g_transform_test)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

class CIFARModel10:
    def __init__(self, model, lr = 0.001, momentum = 0.9, step_size = 0, gamma = 0.01):
        self.m_train_losses = []
        self.m_test_losses = []
        self.m_train_acc = []
        self.m_test_acc = []
        self.m_model=copy.deepcopy(model)
        self.m_optimizer = optim.SGD(self.m_model.parameters(), lr, momentum)
        self.m_criterion = nn.CrossEntropyLoss()
        #self.m_optimizer = optim.Adam(self.m_model.parameters(), lr)
        self.m_optimizer = optim.SGD(self.m_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        #self.adjuster = scheduler.StepLR(self._optimizer, args.epoch_step,gamma=args.gamma)
        #self.m_scheduler = StepLR(self.m_optimizer, step_size=10, gamma=0.5)                                 
        self.load_cifar_data(g_train_set,g_test_set)
        
    def clone_model(self):
        copy.deepcopy(self.m_model)
    
    def load_cifar_data(self,train_set,test_set):
        
        self.m_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                          shuffle=True, num_workers=4)
        self.m_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                         shuffle=False, num_workers=4)
                                         
