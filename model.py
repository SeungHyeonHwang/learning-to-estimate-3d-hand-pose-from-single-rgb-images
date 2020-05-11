# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:41:24 2020

@author: hwang

 - pipeline: HandSegNet + PoseNet + PosePrior

"""
import torch
import torch.nn as nn
import torchvision.models as models

# HandSegNet
class HandSegNet(nn.Module):
	def __init__(self, num_classes=1): 
		super(HandSegNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4, stride=2,padding=1))
        
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1),
			nn.MaxPool2d(kernel_size=4,stride=2,padding=1))
        
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4,stride=2,padding=1))
        
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 2,kernel_size=1,stride=1,padding=0),
    		nn.Upsample(size=(256, 256), mode='bilinear'))
            
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		out = self.layer4(x)
		return out

# PoseNet
class PoseNet(nn.Module):
    def __init__(self, num_classes=21): 
        super(PoseNet, self).__init__()

        self.layer1 = nn.Sequential(
  
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=2,padding=1),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=2,padding=1),
          
        nn.Conv2d(128, 256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),    
        nn.MaxPool2d(kernel_size=4, stride=2,padding=1),
        
        nn.Conv2d(256, 512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=1),
        nn.ReLU())

        self.conv1 = nn.Conv2d(512, 21, kernel_size=1,stride=1,padding=1)
        self.layer2 = nn.Sequential(
            
            nn.Conv2d(533, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),         
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),   
            nn.Conv2d(128,21,kernel_size=1,stride=1,padding=0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(554, 128, kernel_size=7,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),         
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),  
            nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0))
		
    def forward(self, x):
        out1 = self.layer1(x) # 16
        out2 = self.conv1(out1) # 17
        print(out1.shape, out2.shape)
        out3 = torch.cat((out1, out2), dim=1) # 18
        out4 = self.layer2(out3) # 24
        print(out3.shape, out4.shape)
        out5 = torch.cat((out3, out4), dim=1) # 25
        print(out5.shape)
        out = self.layer3(out5) # 31
        print(out.shape)
        return out

    

# PosePrior
class PosePrior(nn.Module):
    def __init__(self, num_classes=1): 
        super(PosePrior, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(21, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Conv2d(32, 32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Conv2d(32, 64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(), 
        nn.Dropout(p=0.2),
        nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),   
        nn.Dropout(p=0.2),
        nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.2))
        
       
        self.fc = nn.Sequential(
        nn.Linear(4 * 4 * 130, 512),
        nn.Dropout(p=0.2),
        nn.Linear(512, 512),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes))
		
    def forward(self, x, hand_side):
        x = self.layer1(x)
        
        x = x.view(x.size(0), -1) # Reshape 
        x = torch.cat((x, hand_side), dim=1) # concat
        
        out = self.fc(x)
                
        return out

    