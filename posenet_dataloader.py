# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:40:49 2020

@author: hwang
"""

import cv2
import os
import numpy as np
import glob
# import pandas as pd
# import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
from scipy import io
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from scipy import ndimage as ndi

def train_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    DATASET_PATH = 'C:/Users/USER/Desktop/hand/data/'
    train_image_dir = os.path.join(DATASET_PATH, 'training', 'color') 
    # val_image_dir = os.path.join(DATASET_PATH, 'images', 'B1Random') 
    label_path = os.path.join(DATASET_PATH, 'training') 

    posenet_dataloader = DataLoader(
        PoseNetDataset(train_image_dir, label_path=label_path, 
                transform=transforms.Compose([
                                              # transforms.ColorJitter(hue=0.1),
                                              transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return posenet_dataloader


class PoseNetDataset(Dataset):
    def __init__(self, image_data_path, label_path=None, transform=None):
        # self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform
        
        # if self.label_path is not None:
            # self.label_matrix = np.load(label_path)
            # self.label_matrix = io.loadmat(label_path) 
        # self.label_matrix = io.loadmat(self.label_path + '/anno_training.mat') 
        # self.label_matrix = np.load(self.label_path+"anno_training.pickle", allow_pickle=True)
        
        # idx = 1000
        # print(label_matrix['frame'+str(idx)])
        # print(self.label_matrix)  
        self.loaded_data = np.load(self.label_path+"/anno_training.pickle", allow_pickle=True)
        
    def __len__(self):
        images = glob.glob(self.image_dir+'/*.png')
        return len(images)
    
    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir + '/'+str(idx).zfill(5)+'.png')
        
        mask_path = os.path.join('C:/Users/USER/Desktop/hand/data/', 'training', 'mask') 
        msk_name = os.path.join(mask_path +'/'+str(idx).zfill(5)+'.png')
        
        new_img = Image.open(img_name).convert('RGB')
        mask_img = Image.open(msk_name)
           
        img = np.array(new_img, 'uint8')
        mask = np.array(mask_img, 'uint8')
        
        mask = np.where(mask>1, 255, 0)
        x_size,y_size,depth=np.shape(img)
        mask[mask>1]=255
        mask[mask<=1]=0
        markers = ndi.label(mask)[0]
        masks = watershed(mask,markers,mask=mask)
          
        num=0
        index=0
        for i in range(1,np.max(masks)+1):
           if((np.sum(mask[masks==i])/255)>num ):
               num=np.sum(mask[masks==i])/255
               index=i
          
          
        mask[masks>index]=0
        mask[masks<index]=0
        lo=np.where(255==mask)
        lo=np.array(lo)
        x_min=np.min(lo[1])
        x_max=np.max(lo[1])
        y_min=np.min(lo[0])
        y_max=np.max(lo[0])
        
        crop_img=img[y_min:y_max,x_min:x_max]
        
        crop_img=cv2.resize(crop_img,(y_size, x_size), interpolation=cv2.INTER_LINEAR)  
        
        img = Image.fromarray(img)
        crop_img = Image.fromarray(crop_img)
        new_img = self.transform(img)
        crop_img = self.transform(crop_img)
        
        # Wrel 
        # self.loaded_data 
        
        # print(self.loaded_data[idx]['xyz']) # 42x3
        # print(self.loaded_data[idx]['uv_vis']) # 42x3
        # print(self.loaded_data[idx]['K']) # 3x3
          
        return new_img, crop_img


        