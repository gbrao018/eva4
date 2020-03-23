from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms

def train_model(model, device, train_loader, loss_fn, optimizer, epoch,train_losses,train_acc, doL1 = 0,doL2 = 0,LAMBDA = 0):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #print('y_pred=',len(y_pred.dataset),'target=',len(target.dataset))
    #loss = F.nll_loss(y_pred, target)
    loss = loss_fn(y_pred, target)
    reg_loss=0
    if (doL1 == 1):
      for p in model.parameters():  
        reg_loss += torch.sum(torch.abs(p.data))
    if (doL2 == 1):
      for p in model.parameters():
        reg_loss += torch.sum(p.data.pow(2))    
    
    loss+=LAMBDA*reg_loss
    
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test_model(model, device, test_loader, loss_fn, test_losses,test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))    
#Global functions
def show_summary(model,input_size = (1, 28, 28)):
    summary(model.m_model, input_size)
    
def run_model(model, device, doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (epoch+start))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, epoch,model.m_train_losses,model.m_train_acc,doL1,doL2,LAMBDA)
        test_model(model.m_model, device, model.m_test_loader,model.m_test_losses,model.m_test_acc)

import matplotlib.pyplot as plt
def draw_accuracy_loss_change_graps(model_0,model_l1,model_l2,model_l1_l2):
    fig, axs = plt.subplots(2,2,figsize=(30,20))
    #print('train_losses=',len(train_losses))
    #print('test_losses=',len(test_losses))

    axs[0,0].plot(model_0.m_test_losses,color='black',label='No Regularization')
    axs[0,0].plot(model_l1.m_test_losses,color='red',label='L1 Regularization')
    axs[0,0].plot(model_l2.m_test_losses,color='blue',label='L2 Regularization')
    axs[0,0].plot(model_l1_l2.m_test_losses,color='green',label='Both L1 and L2 Regularization')
    axs[0,0].set_title("Validation Loss Change")
    axs[0,0].legend(loc="center")

    axs[0,1].plot(model_0.m_test_acc,color='black',label='No Regularization')
    axs[0,1].plot(model_l1.m_test_acc,color='red',label='L1 Regularization')
    axs[0,1].plot(model_l2.m_test_acc,color='blue',label='L2 Regularization')
    axs[0,1].plot(model_l1_l2.m_test_acc,color='green',label='Both L1 and L2 Regularization')
    axs[0,1].set_title("Validation Accuracy Change")
    axs[0,1].legend(loc="center")

    axs[1,0].plot(model_0.m_train_losses,color='black',label='No Regularization')
    axs[1,0].plot(model_l1.m_train_losses,color='red',label='L1 Regularization')
    axs[1,0].plot(model_l2.m_train_losses,color='blue',label='L2 Regularization')
    axs[1,0].plot(model_l1_l2.m_train_losses,color='green',label='Both L1 and L2 Regularization')
    axs[1,0].set_title("Training Loss Change")
    axs[1,0].legend(loc="center")

    axs[1,1].plot(model_0.m_train_acc,color='black',label='No Regularization')
    axs[1,1].plot(model_l1.m_train_acc,color='red',label='L1 Regularization')
    axs[1,1].plot(model_l2.m_train_acc,color='blue',label='L2 Regularization')
    axs[1,1].plot(model_l1_l2.m_train_acc,color='green',label='Both L1 and L2 Regularization')
    axs[1,1].set_title("Training Accuracy Change")
    axs[1,1].legend(loc="center")
