import importlib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
import logging

epoch_size = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig(filename = "/content/dense_mask_loss.log",datefmt='%H:%M:%S',level=logging.DEBUG)

def train(modal, train_loader, optimizer, epoch, preds, labels,last_batch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  correct = 0
  processed = 0

    for batch_idx, input_array in enumerate(train_loader):
        optimizer.zero_grad()
        
        bg = input_array[0].to(device)
        fgbg = input_array[1].to(device)
        concat = torch.cat([bg,fgbg],dim=1)
        depth_img = input_array[2].to(device)
        mask_img = input_array[3].to(device)
        
        # compute output
        starttime = timeit.default_timer()
        out_array = modal(concat)#input to depth and mask modal
        modal_exec_t =  timeit.default_timer() - starttime 

        depth_pred = out_array[0]
        mask_pred = out_array[1]
        
        criteria1 = torch.nn.L1Loss()
        criteria2 = torch.nn.MSELoss()
        
        lossd1 = criteria1(depth_pred,depth_img)
        starttime = timeit.default_timer()
        lossd2 = criteria2(depth_pred,depth_img)
        l2_d_time = timeit.default_timer() - starttime

        starttime = timeit.default_timer()
        ssim_loss_d = 1 - ssim(depth_img,depth_pred) 
        ssim_time = timeit.default_timer() - starttime
        
        grad_loss = gradient_loss(depth_pred,depth_img)
        smooth_l1_loss = torch.nn.SmoothL1Loss()
        
        lossd3 = smooth_l1_loss(depth_pred,depth_img)
        
        starttime = timeit.default_timer()
        l2_mask = criteria2(mask_pred,mask_img)
        l2_m_time = timeit.default_timer() - starttime

        loss2 = smooth_l1_loss(mask_pred,mask_img)
        bce_loss = torch.nn.BCELoss()
        mask_bce_loss = bce_loss(mask_pred,mask_img)

        #-->FINAL loss =  20* lossd2  +   ssim_loss_d +   l2_mask
        loss =  70 * (lossd2  +  mask_bce_loss)
        idx = batch_idx
        if batch_idx < last_batch:
          pred = depth_pred.cpu().detach().numpy()[:,:,:,-1]
          label = depth_img.cpu().detach().numpy()[:,:,:,-1]
      
          labels[:, :, idx] = label
          preds[:, :, idx] = pred[0, :, :]
          rel_error = np.mean(np.abs(preds - labels)/labels)
          #print('\nMean Absolute Relative Error: {:.6f}'.format(rel_error))        
          rmse = np.sqrt(np.mean((preds - labels)**2))
          #print('Root Mean Squared Error: {:.6f}'.format(rmse))

          log10 = np.mean(np.abs(np.log10(preds) - np.log10(labels)))
          #print('Mean Log10 Error: {:.6f}'.format(log10))

          acc = np.maximum(preds/labels, labels/preds)
          delta1 = np.mean(acc < 1.25)
          #print('Delta1: {:.6f}'.format(delta1))

          delta2 = np.mean(acc < 1.25**2)
          #print('Delta2: {:.6f}'.format(delta2))

          delta3 = np.mean(acc < 1.25**3)
        
        starttime = timeit.default_timer()
        loss.backward()
        backprop_exec_t = timeit.default_timer() - starttime
        
        optimizer.step()
        
        desc = f'Loss={loss.item()} Epoch={epoch} Batch_id={batch_idx}  L2-D={lossd2.item():0.6f} L2-M={l2_mask.item():0.6f} BCE-M={mask_bce_loss.item():0.6f} SSIM-D={ssim_loss_d.item():0.6f} MODAL-EXEC-TIME={modal_exec_t:0.3f} BACKPROP-EXEC-TIME={backprop_exec_t:0.3f} L2-DEPTH-TIME={l2_d_time:0.3f} L2-MASK-TIME={l2_m_time:0.3f} SSIM-DEPTH-TIME={ssim_time:0.3f} GRAD-D={grad_loss.item():0.6f} SMOOTH-D={lossd3.item():0.6f} SMOOTH-L1-M={loss2.item():0.6f} RMSE={rmse:0.6f} Meanlog10={log10:0.6f} Acc_D1={delta1:0.6f} Acc_D2={delta2:0.6f} Acc_D3={delta3:0.6f}'
        logging.info(desc)
        pbar.set_description(desc)

        if batch_idx % 30 ==0:
          torch.save(modal.state_dict(), 'densednn2_depth_mask_256.pth')
          torch.save(modal.state_dict(), '/content/gdrive/My Drive/eva-04/densednn2_depth_mask_256.pth')
        
        if batch_idx > (last_batch-3): # save the last batch values
          torch.save(modal.state_dict(), 'densednn32_depth_mask_256.pth')
          torch.save(modal.state_dict(), '/content/gdrive/My Drive/eva-04/densednn32_depth_mask_256.pth')

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)
