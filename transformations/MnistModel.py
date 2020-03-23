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

g_train_transforms = transforms.Compose([
                              #  transforms.Resize((28, 28)),
                              #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                               transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                               # Note the difference between (0.1307) and (0.1307,)
                               ])
g_test_transforms = transforms.Compose([
                              #  transforms.Resize((28, 28)),
                              #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])                               

g_train_set = datasets.MNIST('./data', train = True, download = True, transform = g_train_transforms)
g_test_set = datasets.MNIST('./data', train = False, download = True, transform = g_test_transforms)    
        
class MnistModel:
    def __init__(self, model,dataloader_args, lr = 0.01, momentum = 0.9, step_size = 0, gamma = 0.01):
        self.m_train_losses = []
        self.m_test_losses = []
        self.m_train_acc = []
        self.m_test_acc = []
        self.m_model=copy.deepcopy(model)
        self.m_loss_fn = F.nll_loss()
        self.m_optimizer=optim.SGD(self.m_model.parameters(), lr, momentum)
        #self.m_optimizer = optim.SGD(m_model.parameters(), lr, momentum)
        self.load_mnist_data(dataloader_args,g_train_set,g_test_set)
        
    def clone_model(self):
        copy.deepcopy(self.m_model)
    
    def load_mnist_data(self,dataloader_args,train_set,test_set):
        #cuda = torch.cuda.is_available()
        #print("CUDA Available?", cuda)        
        # For reproducibility
        #torch.manual_seed(SEED)
        #if cuda:
        #    torch.cuda.manual_seed(SEED)
        # dataloader arguments - something you'll fetch these from cmdprmt
        #dataloader_args = dict(shuffle, batch_size, num_workers, pin_memory) if cuda else dict(shuffle = True, batch_size = 64)
        # train dataloader
        self.m_train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        # test dataloader
        self.m_test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

