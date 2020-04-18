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
import random
import math

def create_and_split_dataset(dir):
    """
    Helper Function to make a dataset containing all images in a certain directory
    :param dir: the directory containing the dataset
    :return: images: list of image paths
    """
    train_file_names = []
    train_class_names = []
    test_file_names = []
    test_class_names = []
    classes_dict = dict()
    test_classname_dict = dict()
    train_classname_dict = dict()

    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dire_name = ''
    idx =0
    for root, dirs, fnames in sorted(os.walk(dir)):
      idx = 0
      for fname in fnames:
        if is_image_file(fname):
          idx += 1
          path = os.path.join(root, fname)
          #self.image_paths.append(path)
          dir_name = os.path.basename(os.path.dirname(root))
          #if dir_name not in g_class_to_idx.keys():
            #Not present add it
          #classes_dict[fname] = class_idx
          classes_dict[fname] = g_class_to_idx[dir_name]
          #class_idx = class_idx+1

          if idx > 7:
            test_file_names.append(fname)
            #test_class_names.append(os.path.basename(os.path.dirname(root)))
            test_classname_dict[fname] = os.path.basename(os.path.dirname(root))
          else:
            train_file_names.append(fname)
            #train_class_names.append(os.path.basename(os.path.dirname(root)))
            train_classname_dict[fname] = os.path.basename(os.path.dirname(root))

          idx = idx % 10
          #print(idx)
          #print(fname,'-',os.path.basename(os.path.dirname(root)))
    return  train_file_names, test_file_names, train_classname_dict, test_classname_dict, classes_dict


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    """
    Helper Function to determine whether a file is an image file or not
    :param filename: the filename containing a possible image
    :return: True if file is image file, False otherwise
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
