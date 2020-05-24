import numpy as np
import cv2
import io
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
from PIL import Image
import sys
import torch.utils.data
import timeit
import logging

# This is the common class for both test and train data sets. Based on indexes, this class serves the sample.
#example folder structure

#bg1
  #   /fg_1
  #      /depth
  #           1.jpg
  #           40.jpg  
  #      /mask_
  #      /overlay
  #   /fg_2
  #     /depth
  #     /mask_
  #     /overlay
  
  
#Given an index, we can identify the paricular folder and files 

class BackgroundSubtraction_Dataset(Dataset):

  def __init__(self, root, size, test = False, start= 1,  transform=ToTensor()):
    'Initialization'
    self.m_root = root
    self.m_bg_folder = os.path.join(self.m_root, "bg150")
    self.m_transform = transform
    self.g_bg_dict = dict()
    self.g_overlay_dict = dict()
    self.g_depth_dict = dict()
    self.g_mask_dict = dict()
    logging.basicConfig(filename = "/content/dataloader_timeit.log",datefmt='%H:%M:%S',level=logging.DEBUG)
    self.m_size = size
    self.m_btest = test 
    self.m_start = start
  
  
  #bg1
  #   /fg_1
  #      /depth
  #           1.jpg
  #           40.jpg  
  #      /mask_
  #      /overlay
  #   /fg_2
  #     /depth
  #     /mask_
  #     /overlay
  
     
  def __len__(self):
    'Denotes the total number of samples'
    #return len(self.m_files)
    #return len(self.m_size)
    return self.m_size

  def __getitem__(self, idx):

    starttime = timeit.default_timer()

    #index = idx + 4001
    if self.m_btest == True:
      index = idx + self.m_start
    else:
      index = idx + 1

    starttime = timeit.default_timer()
  
    bg_folder ="bg"
    fgbg_folder = "bg"
    fg_folder ="fg_"

    overlay_folder = "overlay"
    mask_folder = "mask"
    depth_folder = "depth"

    bg_index = index // 4000 # each fg can hold till 4000
    if index % 4000 != 0:
      bg_index = bg_index + 1

    # each fg can hold till 40
    fg_index = index // 40
    if index % 40 != 0:
      fg_index = fg_index + 1
  
    fg_index = fg_index % 100 # max cap for fg_index
    if fg_index == 0:
      fg_index =  100 

    file_index = index % 40
    if file_index == 0:
        file_index = 40
    
    fgbg_folder = fgbg_folder + str(bg_index)
    fg_folder = fg_folder + str(fg_index)
    filename = str(file_index)+".jpg"
    #print("filename=",filename)
    bg_path = os.path.join(self.m_root,bg_folder)
    bg_file_name = bg_folder + str(bg_index) + ".jpg"
    bg_file = os.path.join(bg_path,bg_file_name) # bg file
  
    # get fgbg image paths for overlay,depth,mask
    fgbg_path = os.path.join(self.m_root, fgbg_folder)
    fg_folder = os.path.join(fgbg_path,fg_folder)
    overlay_folder = os.path.join(fg_folder,overlay_folder)
    depth_folder = os.path.join(fg_folder,depth_folder) 
    mask_folder = os.path.join(fg_folder,mask_folder)

    overlay_file = os.path.join(overlay_folder,filename) # overlay file
    depth_file = os.path.join(depth_folder,filename) # depth file
    mask_file = os.path.join(mask_folder,filename) # mask file

    # Take all images
    #LA -> luninosity and Alpha channels
    #find the bg file:
    bg_image = Image.open(bg_file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
    bg_image = bg_image.resize((256,256))
    if self.m_transform:
      bg_image = self.m_transform(bg_image)
    

    overlay_image = Image.open(overlay_file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
    overlay_image = overlay_image.resize((256,256))
    if self.m_transform:
      overlay_image = self.m_transform(overlay_image)
    
    depth_image = Image.open(depth_file) # even grey scale images will be converted to RGB with 3 channels
    depth_image = depth_image.resize((256,256))
    if self.m_transform:
      depth_image = self.m_transform(depth_image)
    
    mask_image = Image.open(mask_file)
    mask_image = mask_image.resize((256,256))
    if self.m_transform:
      mask_image = self.m_transform(mask_image)
    
    # Take all images
    
    input_array = []
    input_array.append(bg_image)
    input_array.append(overlay_image)
    input_array.append(depth_image)
    input_array.append(mask_image)
    input_array.append(bg_file)
    input_array.append(overlay_file)
    input_array.append(depth_file)
    input_array.append(mask_file)
    load_time = timeit.default_timer() - starttime
    
    if self.m_btest == True:
      desc = f' bg-file={bg_file} fgbg_file={overlay_file} depth_file={depth_file}  LOAD_TIME={load_time:0.3f}'
      logging.info(desc)
    
    #print("dict sizes,",len(g_bg_dict),len(g_overlay_dict),len(g_depth_dict),len(g_mask_dict))  
    return input_array


  def create_indexes(self, transform = transforms.ToTensor()):
    
    for idx in range(0,30000):
      starttime = timeit.default_timer()
      index = idx + 1

      bg_folder ="bg"
      fgbg_folder = "bg"
      fg_folder ="fg_"

      overlay_folder = "overlay"
      mask_folder = "mask"
      depth_folder = "depth"

      bg_index = index // 4000 # each fg can hold till 4000
      if index % 4000 != 0:
        bg_index = bg_index + 1

      # each fg can hold till 40
      fg_index = index // 40
      if index % 40 != 0:
        fg_index = fg_index + 1
    
      fg_index = fg_index % 100 # max cap for fg_index
      if fg_index == 0:
        fg_index =  100 

      file_index = index % 40
      if file_index == 0:
          file_index = 40
      
      fgbg_folder = fgbg_folder + str(bg_index)
      fg_folder = fg_folder + str(fg_index)
      filename = str(file_index)+".jpg"
      #print("filename=",filename)
      root = self.m_root
      bg_path = os.path.join(root,bg_folder)
      bg_file_name = bg_folder + str(bg_index) + ".jpg"
      bg_file = os.path.join(bg_path,bg_file_name) # bg file
    
      # get fgbg image paths for overlay,depth,mask
      fgbg_path = os.path.join(root, fgbg_folder)
      fg_folder = os.path.join(fgbg_path,fg_folder)
      overlay_folder = os.path.join(fg_folder,overlay_folder)
      depth_folder = os.path.join(fg_folder,depth_folder) 
      mask_folder = os.path.join(fg_folder,mask_folder)

      overlay_file = os.path.join(overlay_folder,filename) # overlay file
      depth_file = os.path.join(depth_folder,filename) # depth file
      mask_file = os.path.join(mask_folder,filename) # mask file

      # Take all images
      #LA -> luninosity and Alpha channels
      #find the bg file:
      bg_image = self.g_bg_dict.get(index,None)

      if bg_image is None:
        bg_image = Image.open(bg_file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
        bg_image = bg_image.resize((256,256))
        if transform:
          bg_image = transform(bg_image)
        #print('saving index into bg_dict',index)
        self.g_bg_dict[index] = bg_image
      else:
        print('File picked from memory',index)
      


      overlay_image = self.g_overlay_dict.get(index,None)
      if overlay_image is None:
        overlay_image = Image.open(overlay_file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
        overlay_image = overlay_image.resize((256,256))
        if transform:
          overlay_image = transform(overlay_image)
        self.g_overlay_dict[index] = overlay_image
      else:
        print('File picked from memory',index)
      
      depth_image = self.g_depth_dict.get(index,None)
      if depth_image is None:  
        #depth_image = Image.open(depth_file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
        #mask_image = Image.open(mask_file).convert('RGB')
        depth_image = Image.open(depth_file) # even grey scale images will be converted to RGB with 3 channels
        depth_image = depth_image.resize((256,256))
        if transform:
          depth_image = transform(depth_image)
        self.g_depth_dict[index] = depth_image
      else:
        print('File picked from memory',index)
      

      mask_image = self.g_mask_dict.get(index,None)
      if mask_image is None:  
        mask_image = Image.open(mask_file)
        mask_image = mask_image.resize((256,256))
        if transform:
          mask_image = transform(mask_image)
        self.g_mask_dict[index] = mask_image
      else:
        print('File picked from memory',index)
      
      load_time = timeit.default_timer() - starttime
      desc = f' bg-file={bg_file} fgbg_file={overlay_file} depth_file={depth_file}  LOAD_TIME={load_time:0.3f}'
      logging.info(desc)
      #return self.g_bg_dict, g_overlay_dict, g_depth_dict, g_mask_dict

