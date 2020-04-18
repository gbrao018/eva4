import numpy as np
import cv2
import io
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
import torchvision.datasets.folder

# This is the common class for both test and train data sets. Based on indexes, this class serves the sample.
class ImagenetDataset(Dataset):
  def __init__(self, root, files, classname_dict, class_dict, transform=ToTensor()):
    'Initialization'
    self.m_root = root
    self.m_class_dict = class_dict
    self.m_classname_dict = classname_dict
    #self.m_file_path_dict = file_path_dict
    self.m_files = files
    self.m_transform = transform

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.m_files)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    file_name = self.m_files[index]
    class_name = self.m_classname_dict[file_name]
    # Load data and get label
    dir_path = os.path.join(self.m_root, class_name)
    image_dir = os.path.join(dir_path, 'images')
    image_name = os.path.join(image_dir,file_name)
    image = Image.open(image_name).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
    try:
      if self.m_transform:
        image = self.m_transform(image)
    except:
      print('Error while transform:', image_name, image.shape)
    return image, self.m_class_dict[file_name]
