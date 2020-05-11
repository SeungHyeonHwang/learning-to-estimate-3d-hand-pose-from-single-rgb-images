# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:40:49 2020

@author: hwang
"""

import cv2
import os
import numpy as np
import glob
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy import io

def train_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    DATASET_PATH = 'C:/Users/USER/Desktop/hand/data/'
    train_image_dir = os.path.join(DATASET_PATH, 'training', 'color') 
    label_path = os.path.join(DATASET_PATH, 'training', 'mask') 

    handseg_dataloader = DataLoader(
        HandDataset(train_image_dir, label_path=label_path, 
                transform=transforms.Compose([
                                              transforms.ColorJitter(hue=0.1),
                                              transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return handseg_dataloader


class HandDataset(Dataset):
    def __init__(self, image_data_path, label_path=None, transform=None):
        # self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform
        
        # if self.label_path is not None:
            # self.label_matrix = np.load(label_path)
            # self.label_matrix = io.loadmat(label_path) 
          
    def __len__(self):
        images = glob.glob(self.image_dir+'/*.png')
        return len(images)
    def __getitem__(self, idx):
        # i=0
        img_name = os.path.join(self.image_dir + '/'+str(idx).zfill(5)+'.png')
        # print(img_name)
        # i+=1
        mask_name = os.path.join(self.label_path + '/'+str(idx).zfill(5)+'.png')
        new_img = Image.open(img_name).convert('RGB')
        mask_img = Image.open(mask_name)
        
        # img_numpy = np.array(new_img, 'uint8')
        
        
        # mask_img = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        # mask_img = mask_img.astype(np.uint8)
        # mask_img = ((mask_img - mask_img.min()) * (1/(mask_img.max() - mask_img.min()) * 255)).astype('uint8')
        # print(mask_img)
        # mm = Image.fromarray(mask_img)
        msk_np = np.array(mask_img, 'uint8')
        # msk_max = msk_np.max()
        
        # if msk_max > 17 : 
        mask_img = np.where(msk_np>1, 1, 0)
        
        # msk_np = ((msk_np - msk_np.min()) * (1/(msk_np.max() - msk_np.min()) * 255)).astype('uint8')
        # print(msk_np.max())
        
        # mask_img = np.where(mm>=mm.max()-21, 1, 0)
        # mask_img = np.where(msk_np>=150, 1, 0)
        
        # print(mask_img.max())
        mask_img = Image.fromarray(mask_img)
        
        
        # random crop 
        i, j, h, w = transforms.RandomCrop.get_params(mask_img, output_size=(256, 256)) 
        new_img = transforms.functional.crop(new_img, i, j, h, w) 
        mask_img = transforms.functional.crop(mask_img, i, j, h, w)

        # mask_img = mask_img.resize((256, 256))
        
        trans = transforms.ToTensor()
        
        msk = trans(mask_img)
        if self.transform:

            new_img = self.transform(new_img)
            # print(mask_img)
            # mask_img = self.transform(mask_img)
            
            
        return new_img, msk
        # print(new_img.shape)
        # mask_img = rgb2gray(new_img)
        # mask_img = Image.open(mask_img)
        # mask_img = self.transform(mask_img)
        
        # if self.label_path is not None:
        #     hand_para = self.label_matrix['handPara'][...,idx]
        #     hand_para = torch.tensor(hand_para) # here, we will use only one label among multiple labels.
        #     hand_side = [0,1]
        #     hand_side = torch.tensor(hand_side)
        #     # print(len(new_img))
        #     # print(hand_para.shape)
        #     return new_img, mask_img, hand_para, hand_side
        # else:
        #     return new_img, hand_para, [0,1]
        
        