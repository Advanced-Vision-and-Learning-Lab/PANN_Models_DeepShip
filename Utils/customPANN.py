# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018
Generate histogram layer
@author: jpeeples
"""

import torch
import torch.nn as nn
import numpy as np

class CustomPANN(nn.Module):
    def __init__(self,model):

        # inherit nn.module
        super(CustomPANN, self).__init__()
        self.fc=model.fc_audioset
        model.fc_audioset=nn.Sequential()
        self.backbone=model
        
    def forward(self,x):        
        #extract features from PANN model
        x=self.backbone(x)
        # pass the extracted features to the output layer 
        x=self.fc(x)
        
        return x
        
    
    
         
     
        
        
        
        
        
    
