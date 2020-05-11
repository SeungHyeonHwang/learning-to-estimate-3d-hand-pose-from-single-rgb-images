# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:24:17 2020

@author: hwang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
from model import PosePrior, HandSegNet, PoseNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from handseg_dataloader import train_dataloader

def to_np(t):
    return t.cpu().detach().numpy()

def lr_scheduler(optimizer, curr_iter):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
 
    if curr_iter == 30000:
        optimizer.param_groups[0]['lr'] *= 0.1
    if curr_iter == 40000:
        optimizer.param_groups[0]['lr'] *= 0.1

def createFolder(directory):
    try :
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error : " + directory)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--mode', type=str, default='train')

    # custom args
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    
    # Model init
    device = 0
    hand_seg_net = HandSegNet().to(device)
    criterion = nn.CrossEntropyLoss()    
    learning_rates = 1e-5
    optimizer = torch.optim.Adam(hand_seg_net.parameters(), lr=args.learning_rate)

    hand_seg_net.train()
    dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers)
    curr_lr = args.learning_rate
    print("Initializing Training!")
    
    save_path = 'C:/Users/USER/Desktop/hand/model_save/'
    createFolder(save_path)
    
    iteration = 0
    
    min_loss = 9999
    for epoch_idx in range(1, args.epochs + 1):
        total_loss = 0
        total_correct = 0 

        losses = 0
        for batch_idx, (image, mask) in enumerate(dataloader):
            iteration+=1

            x = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output_prob = hand_seg_net.forward(x)
            
            loss = criterion(output_prob, torch.squeeze(mask).long()) 
            loss.backward()
            lr_scheduler(optimizer, iteration)
            optimizer.step()
                               
            losses += loss
            if (batch_idx+1)%(args.log_interval) == 0 : 
                print("Epoch {}/{}   Batch {}/{}   loss {:2.4f} ".format(epoch_idx, args.epochs, (batch_idx+1), len(dataloader), losses/(batch_idx+1)))
                if min_loss >= losses/(batch_idx+1) :
                    min_loss = losses/(batch_idx+1)
                    print("min loss : %f " %(min_loss))
                    
                    torch.save(hand_seg_net.state_dict(), save_path+'handseg.pth')
                    
                    # plt.figure()
                    # img = to_np(image[0,...])
                    # plt.imshow(np.transpose(img, (1,2,0)), cmap='brg')
                    
                    # plt.figure()      
                    # msk = to_np(mask[0,0,:,:])
                    # plt.imshow(msk, cmap='gray')
                               
                    # plt.figure()           
                    # pr = to_np(output_prob[0,1,:,:])
                    # plt.imshow(pr, cmap='gray')                                        
                    # plt.show()            
 