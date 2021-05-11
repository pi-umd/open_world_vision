import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle
import skimage.transform 
import math

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms


def createMontage(imList, dims, times2rot90=0):
    '''
    imList isi N x HxWx3
    making a montage function to assemble a set of images as a single image to display
    '''
    imy, imx, k = dims
    rows = round(math.sqrt(k))
    cols = math.ceil(k/rows)
    imMontage = np.zeros((imy*rows, imx*cols, 3))
    idx = 0
    
    y = 0
    x = 0
    for idx in range(k):
        imMontage[y*imy:(y+1)*imy, x*imx:(x+1)*imx, :] = imList[idx, :,:,:] #np.rot90(imList[:,:,idx],times2rot90)
        if (x+1)*imx >= imMontage.shape[1]:
            x = 0
            y += 1
        else:
            x+=1
    return imMontage



class OWV_dataset(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        
        if self.set_name=='train':
            self.transform = transforms.Compose([                
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])            
        else:
            self.set_name = 'val'
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        self.imageList = imageList
        self.labelList = labelList
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.open(curImage).convert('RGB')        
        curImage = self.transform(curImage)        
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return curImage, curLabel