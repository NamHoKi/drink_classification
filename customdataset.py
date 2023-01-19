import torch
import os
import glob

from torch.utils.data import Dataset

class my_dataset(Dataset) :
  def __init__(self, path, transform=None) :
    self.all_data_path = glob.glob(os.path.join(path, '*', '*.png'))
    self.transform = transform
    
    
