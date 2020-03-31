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
import numpy as np

def train_model(model, device, train_loader, optimizer, epoch,train_losses,train_acc,criteria, doL1 = 0,doL2 = 0,LAMBDA = 0):
  print('L1=',doL1,';L2=',doL2,';LAMBDA=',LAMBDA,'epoch=',epoch)
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)
    #print('data=',len(data),';target=',len(target))

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #print('y_pred=',len(y_pred.dataset),'target=',len(target.dataset))
    #loss = F.nll_loss(y_pred, target)
    #criteria = nn.CrossEntropyLoss()
    loss = criteria(y_pred, target) 
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
    

def test_model(model, device, test_loader,test_losses,test_acc,criteria, correct_samples, incorrect_samples, sample_count=30, last_epoch=False):
    model.eval()
    test_loss = 0
    correct = 0
    #criteria = nn.CrossEntropyLoss()
            
    with torch.no_grad():
        for data, target in test_loader:
            img_batch = data
            data, target = data.to(device), target.to(device)
            #print('data=',len(data),';target=',len(target))
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #test_loss += criteria(output, target).item()
            test_loss += criteria(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            result = pred.eq(target.view_as(pred))
            if last_epoch:
                print('last_epoch=',last_epoch)
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
                    elif list(result)[i] and len(correct_samples) < sample_count:
                        correct_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
            correct += result.sum().item()
            #correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))    
#Global functions
def show_summary(model,input_size = (1, 28, 28)):
    summary(model.m_model, input_size)
    
def run_model(model, device, criteria = F.nll_loss, doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, epoch,model.m_train_losses,model.m_train_acc,criteria,doL1,doL2,LAMBDA)
        test_model(model.m_model, device, model.m_test_loader,model.m_test_losses,model.m_test_acc,criteria)

def run_model_with_entropy(model, device, criteria = nn.CrossEntropyLoss(), doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, epoch,model.m_train_losses,model.m_train_acc,criteria,doL1,doL2,LAMBDA)
        #model.m_scheduler.step()
        last_epoch = False
        if(epoch == (EPOCHS-1)):
            last_epoch = True
        
        test_model(model.m_model, device, model.m_test_loader, model.m_test_losses, model.m_test_acc, model.m_criterion, model.m_correct_samples, model.m_incorrect_samples, 30, last_epoch)

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

def unnormalize(image, mean, std, out_type='array'):
    """Un-normalize a given image.
    
    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            If `array` then ndarray is returned else if
            `tensor` then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))
    
    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'array':
        return normal_image
    return None  # No valid value given


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.
    Args:
        tensor: Tensor to be converted.
    """
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.
    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))    