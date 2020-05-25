from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
#import cv2_utils
from torchvision import transforms
#w = torch.load("/content/dense32_depth_mask_128.pth")
#w = torch.load("/content/verygood_128_dw70_sum_100k_lre5_10.pth")
#modal.load_state_dict(w)
import random
import time  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def test(modal, val_loader, epoch, preds, labels, last_batch):
  with torch.no_grad():
    for input_array1 in val_loader:
      count = count + 1
      bg = input_array1[0].to(device)
      fgbg = input_array1[1].to(device)
      depth_img = input_array1[2].to(device)
      mask_img = input_array1[3].to(device)

      concat = torch.cat([bg,fgbg],dim=1)
      output = modal(concat)
  
      bg = bg.cpu()
      fgbg = fgbg.cpu()
      
      depth_pred = output[0].cpu()
      mask_pred = output[1].cpu()
      
      criteria2 = torch.nn.MSELoss()
      lossd2 = criteria2(depth_pred,depth_img)  
      bce_loss = torch.nn.BCELoss()
      mask_bce_loss = bce_loss(mask_pred,mask_img)

      #-->FINAL loss =  20* lossd2  +   ssim_loss_d +   l2_mask
      idx = batch_idx
      if batch_idx < last_batch:
        pred = depth_pred.cpu().detach().numpy()[:,:,:,-1]
        label = depth_img.cpu().detach().numpy()[:,:,:,-1]
  
        labels[:, :, idx] = label
        preds[:, :, idx] = pred[0, :, :]
        
        acc = np.maximum(preds/labels, labels/preds)
        delta1 = np.mean(acc < 1.25)
        #print('Delta1: {:.6f}'.format(delta1))
        delta2 = np.mean(acc < 1.25**2)
        #print('Delta2: {:.6f}'.format(delta2))
        delta3 = np.mean(acc < 1.25**3)
        
        desc = f'Loss={loss.item()} Epoch={epoch} Batch_id={batch_idx}  L2-DEPTH={lossd2.item():0.6f} BCE-MASK={mask_bce_loss.item():0.6f}  Acc-Delat1={delta1:0.6f} Acc-Delat2={delta2:0.6f} Acc-Delat3={delta3:0.6f}'
        logging.info(desc)
        pbar.set_description(desc)
    