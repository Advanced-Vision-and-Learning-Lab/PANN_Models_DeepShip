#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:55:14 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division
import numpy as np

import torch

import pdb

import torch.nn.functional as F
import os

import lightning as L
import torchmetrics
from Utils.Network_functions import initialize_model

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, Dataset, pretrained_loaded, run_number):
        super().__init__()
        
        self.learning_rate = Params['lr']
        self.run_number = run_number

        self.model_ft, self.mel_extractor = initialize_model(
            model_name, 
            use_pretrained=Params['use_pretrained'], 
            feature_extract=Params['feature_extraction'], 
            num_classes=num_classes,
            pretrained_loaded=pretrained_loaded 
        )

        
        # self.test_preds = []
        # self.test_labels = []
        # self.test_features = []

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average='weighted')

        self.save_hyperparameters()

    def forward(self, x):
        # Extract mel spectrogram if not PANN model
        x = self.mel_extractor(x)
        # features are from the feature layer (backbone)
        features, y_pred = self.model_ft(x)
        return features, y_pred


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        features, y_pred = self(x) 
        loss = F.cross_entropy(y_pred, y)
        
        #self.train_acc.update(y_pred, y)
        
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)
        
        return loss

    #def on_train_epoch_end(self):
    #    self.log('train_acc_epoch', self.train_acc.compute())
    #    self.train_acc.reset()   


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        features, y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
        
    #    self.val_acc.update(y_pred, y)
        
        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
    
        return val_loss
 
    #def on_validation_epoch_end(self):
    #    self.log('val_acc_epoch', self.val_acc.compute())
    #    self.val_acc.reset()   
 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        features, y_pred = self(x)
        test_loss = F.cross_entropy(y_pred, y)
        
        #self.test_acc.update(y_pred, y)
        #self.test_f1.update(y_pred, y)
        
        self.test_acc(y_pred, y)
        self.test_f1(y_pred, y)  
    
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)  
    
        # Store predictions, true labels, and features
        # self.test_preds.append(y_pred.cpu())
        # self.test_labels.append(y.cpu())
        # self.test_features.append(features.cpu())
    
        return test_loss
    
    #def on_test_epoch_end(self):
    #    self.log('test_acc_epoch', self.test_acc.compute())
    #    self.val_acc.reset()   
    #   
    #    self.log('test_f1_epoch', self.test_f1.compute())
    #    self.test_f1.reset()   
    #    self.save_features("test", self.hparams.model_name, self.run_number)
    

    
    # def save_features(self, phase, model_name, run_number):
    #     os.makedirs("features", exist_ok=True)
    #     features_file_path = f"features/{model_name}_{phase}_run{run_number}_features.pth"
    #     #labels_file_path = f"features/{model_name}_{phase}_run{run_number}_labels.pth"
    #     #preds_file_path = f"features/{model_name}_{phase}_run{run_number}_preds.pth"
    
    #     # Convert lists to tensors
    #     features_tensor = torch.cat(self.test_features)
    #     labels_tensor = torch.cat(self.test_labels)
    #     preds_tensor = torch.cat(self.test_preds).argmax(dim=1)
    
    #     # Save tensors
    #     torch.save({'features': features_tensor, 'labels': labels_tensor, 'preds': preds_tensor}, features_file_path)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

