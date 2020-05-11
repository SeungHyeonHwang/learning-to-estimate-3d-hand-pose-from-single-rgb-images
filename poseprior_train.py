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
from poseprior_dataloader import train_dataloader

def to_np(t):
    return t.cpu().detach().numpy()

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
    
    
    # Dataset begin
    
    # SEM_train = SEMDataTrain(
    #     '../data/train/images', '../data/train/masks')

    # # TO DO: finish test data loading
    # SEM_test = SEMDataTest(
    #     '../data/test/images/')
    # SEM_val = SEMDataVal(
    #     '../data/val/images', '../data/val/masks')
    # # Dataset end

    # # Dataloader begins
    # SEM_train_load = \
    #     torch.utils.data.DataLoader(dataset=SEM_train,
    #                                 num_workers=16, batch_size=2, shuffle=True)
    # SEM_val_load = \
    #     torch.utils.data.DataLoader(dataset=SEM_val,
    #                                 num_workers=3, batch_size=1, shuffle=True)

    # SEM_test_load = \
    #     torch.utils.data.DataLoader(dataset=SEM_test,
    #                                 num_workers=3, batch_size=1, shuffle=False)
    
    # Dataloader end

    # Data Processing for each model
    
    # handseg
    # raw -> 256x256x3 으로 crop 
    # no data aug.
    

    # Model init
    device = 0

    pose_prior_net = PosePrior(num_classes=3).to(device)
    view_point_net = PosePrior(num_classes=63).to(device)


    # pre-train
    
    # read from pickle file
    # file_name = '*.pickle'
    # with open(file_name, 'rb') as fi:
    #     weight_dict = pickle.load(fi)
    #     weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
        
    # keys = [k for k, v in weight_dict.items() if 'HandSegNet' in k]
    # keys.sort()
        
    # for name, module in handsegnet.named_children():
    #     key = 'HandSegNet/{0}/'.format(name)
    #     if key + 'biases' in weight_dict:
    #         b = torch.Tensor(weight_dict[key + 'biases'])
    #         w = torch.Tensor(weight_dict[key + 'weights'])
    #         w = w.permute((3, 2, 0, 1))
    #         module.weight.data = w
    #         module.bias.data = b
    

    # Loss function
    criterion = nn.MSELoss()   
    learning_rates = 1e-4
    optimizer_prior = torch.optim.Adam(posenet.parameters(), lr=args.learning_rate)
    optimizer_view = torch.optim.Adam(posenet.parameters(), lr=args.learning_rate)

    pose_prior_net.train()
    view_point_net.train()
    
    dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers)

    curr_lr = args.learning_rate
    print("Initializing Training!")
    save_path = 'C:/Users/USER/Desktop/hand/model_save/'
    createFolder(save_path)
    
    iteration = 0
    
    min_loss = 9999
    # losses = 0
    for epoch_idx in range(1, args.epochs + 1):
        losses = 0
        for batch_idx, (score_map, hand_side, labels) in enumerate(dataloader):

            x = score_map.to(device)
            h = hand_side.to(device)
            wrel = labels.to(device)
            
            optimizer_prior.zero_grad()
            optimizer_view.zero_grad()
            
            wc_prob = pose_prior_net.forward(x)
            R_prob = view_point_net.forward(x)
            
            
            R_prob = get_rot(R_prob)
            wrel_prob = torch.matmul(wc_prob, R_prob)
            
            loss1 = criterion(wrel_prob, wrel) 
          
            loss = loss1 + loss2
            loss.backward()
            lr_scheduler(optimizer, iteration)
            optimizer.step()                                





            if (batch_idx+1)%(args.log_interval) == 0 : 
                print("Epoch {}/{}   Batch {}/{}   loss {:2.4f} ".format(epoch_idx, args.epochs, (batch_idx+1), len(dataloader), losses/(batch_idx+1)))
                if min_loss >= losses/(batch_idx+1) :
                    min_loss = losses/(batch_idx+1)
                    print("min loss : %f " %(min_loss))
                    torch.save(posenet.state_dict(), save_path+'posenet.pth')

            plt.figure()
            img = to_np(image[0,...])
            plt.imshow(np.transpose(img, (1,2,0)), cmap='brg')
            
            plt.figure()      
            score_map = to_np(score_map[0,...])          
            plt.imshow(np.transpose(score_map, (1,2,0)), cmap='brg')   
            
            plt.show()            
      