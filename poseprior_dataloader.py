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

def train_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    DATASET_PATH = 'C:/Users/USER/Desktop/hand/data/'
    train_image_dir = os.path.join(DATASET_PATH, 'training', 'color') 
    label_path = os.path.join(DATASET_PATH, 'training') 

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
        

        self.label_matrix = np.load(self.label_path+"anno_training.pickle", allow_pickle=True)

    def __len__(self):
        images = glob.glob(self.image_dir+'/*.png')
        return len(images)
    
    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir + '/'+str(idx).zfill(5)+'.png')
        score_map = Image.open(img_name)

        if msk_max > 17 : 
            hand_side = [1,0]
        else : 
            hand_side = [0,1]
            
        # Wrel 
        # self.loaded_data 
        
        # print(self.loaded_data[idx]['xyz']) # 42x3
        # print(self.loaded_data[idx]['uv_vis']) # 42x3
        # print(self.loaded_data[idx]['K']) # 3x3
            
        score_map = Image.fromarray(score_map)
        
        trans = transforms.ToTensor()
        hand_side = torch.tensor(hand_side)

        return score_map, hand_side
