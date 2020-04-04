from __future__ import print_function
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchsummary import summary
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.optim as optim
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, PadIfNeeded, Cutout, VerticalFlip, Rotate
from albumentations.pytorch import ToTensor
import albumentations as alb

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
class AlbumentationPILImageDataset(Dataset):
    def __init__(self, image_list, augmentation):
        self.m_data_dir = image_list
        self.m_augmentation = augmentation
         
    def __len__(self):
        return (len(self.m_data_dir))
    
    def __getitem__(self, idx):
        img, label = self.m_data_dir[idx]
        
        if self.m_augmentation:
            # Convert PIL image to numpy array
            image_np = np.array(img)
            # Apply transformations
            img = self.m_augmentation(image=image_np)['image']
            # Convert numpy array to PIL Image
            #image = Image.fromarray(augmented['image'])
        return img, label
        
g_alb_pil_transform_train = Compose([
    PadIfNeeded(32,32),
    RandomCrop(32,32), 
    HorizontalFlip(),
    VerticalFlip(),
    Rotate(),
    Cutout(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ToTensor()
])


g_alb_pil_transform_test = Compose([
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ToTensor()
])



g_train_set = datasets.CIFAR10(root='./data', train=True, download=True)

g_test_set = datasets.CIFAR10(root='./data', train=False, download=True)

def mean():
    return tuple(np.mean(g_train_set.data, axis=(0, 1, 2)) / 255)
def std():
    return tuple(np.std(g_train_set.data, axis=(0, 1, 2)) / 255)

g_alb_train_set = AlbumentationPILImageDataset(image_list = g_train_set, augmentation = g_alb_pil_transform_train)
g_alb_test_set = AlbumentationPILImageDataset(image_list = g_test_set, augmentation = g_alb_pil_transform_test)

        
class CIFARModel10_Alb:
    def __init__(self, model, lr = 0.001, momentum = 0.9, step_size = 0, gamma = 0.01):
        self.m_train_losses = []
        self.m_test_losses = []
        self.m_train_acc = []
        self.m_test_acc = []
        self.m_model=copy.deepcopy(model)
        #self.m_optimizer=optim.SGD(self.m_model.parameters(), lr, momentum)
        self.m_criterion = nn.CrossEntropyLoss()
        #self.m_optimizer = optim.Adam(self.m_model.parameters(), lr)
        self.m_optimizer = optim.SGD(self.m_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        #self.adjuster = scheduler.StepLR(self._optimizer, args.epoch_step,gamma=args.gamma)
        self.m_scheduler = StepLR(self.m_optimizer, step_size=10, gamma=0.5)
        self.m_incorrect_samples = []
        self.m_correct_samples = []
        self.load_cifar_data(g_alb_train_set,g_alb_test_set)
        
    def clone_model(self):
        copy.deepcopy(self.m_model)
    
    def load_cifar_data(self,train_set,test_set):
        
        self.m_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                          shuffle=True, num_workers=4)
        self.m_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                         shuffle=False, num_workers=4)
    
                                         
