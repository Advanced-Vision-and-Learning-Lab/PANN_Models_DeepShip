# -*- coding: utf-8 -*-
"""
Functions to generate model and train/validate/test
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from Utils.Histogram_Model import HistRes
from barbar import Bar
from .pytorchtools import EarlyStopping
from Utils.TDNN import TDNN
import pdb
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from Utils.PANN_models import Cnn14



#from src.models import ASTModel
#audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'



def train_model(model, dataloaders, criterion, optimizer, device,feature_extraction_layer,
                saved_bins=None, saved_widths=None, histogram=True,
                num_epochs=25, scheduler=None, dim_reduced=True):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []

    early_stopping = EarlyStopping(patience=10, verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = np.inf
    valid_loss = best_loss
    print('Training Model...')
    

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode 
                feature_extraction_layer.train()
            else:
                model.eval()   # Set model to evaluate mode
                feature_extraction_layer.eval()
            
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                
                inputs = inputs.to(device)

                labels = labels.to(device)
                #index = index.to(device)

    
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    
                    #Pass through feature layer 
                    features = feature_extraction_layer(inputs)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects.item() / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                if(histogram):
                    if dim_reduced:
                        #save bins and widths
                        saved_bins[epoch+1,:] = model.module.histogram_layer[-1].centers.detach().cpu().numpy()
                        saved_widths[epoch+1,:] = model.module.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                    else:
                        # save bins and widths
                        saved_bins[epoch + 1, :] = model.module.histogram_layer.centers.detach().cpu().numpy()
                        saved_widths[epoch + 1, :] = model.module.histogram_layer.widths.reshape(
                            -1).detach().cpu().numpy()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                valid_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            print()
    
        #Check validation loss
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            print()
            break
     
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return losses as dictionary
    train_loss = train_error_history
    
    val_loss = val_error_history
 
    #Return training and validation information
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_loss,'train_acc_track': train_acc_history, 
                  'train_error_track': train_loss,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
     
def test_model(dataloader,model,feature_extraction_layer,criterion,device):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0.0
    model.eval()
    feature_extraction_layer.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels, index) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
            # Forward pass for logits of network
            features = feature_extraction_layer(inputs)
            outputs = model(features)
            loss = criterion(outputs, labels)
           
            #Get predictions for test data
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
        
            # Running statistics for classification metrics
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * inputs.size(0)
            
    test_loss = running_loss / (len(dataloader.sampler))
    test_acc = running_corrects.item() / (len(dataloader.sampler))
    
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc*100,2),
                'test_loss': test_loss}
    
    return test_dict


import os
import requests
def download_weights(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading weights from {url} to {destination}...")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"Weights already exist at {destination}.")

weights_url = "https://zenodo.org/record/3960540/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
weights_path = "./PANN_Weights/Cnn14_16k_mAP=0.438.pth"

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
    
def initialize_model(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=5,
                     feat_map_size=4, TDNN_feats=1, input_feature='STFT',RGB=True,
                     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    
    
    #If TDNN model, only use 1 feature channel
    if model_name == "TDNN":
        RGB = False
 

    #If TDNN model, only use 1 feature channel
    if model_name == "AST":
        RGB = False    

    
    #Intialize feature layer
    feature_layer= Feature_Extraction_Layer(input_feature=input_feature,
                                             RGB=RGB)

    model_ft = None
    input_size = 0
    
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        pass

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(weights='DEFAULT')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "resnet50_wide":
            model_ft = models.wide_resnet50_2(weights='DEFAULT')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "resnet50_next":
            model_ft = models.resnext50_32x4d(weights='DEFAULT')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "densenet121":
            model_ft = models.densenet121(weights='DEFAULT',memory_efficient=True)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "efficientnet":
            model_ft = models.efficientnet_b0(weights='DEFAULT')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "regnet":
            model_ft = models.regnet_x_400mf(weights='DEFAULT')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "TDNN": 
            model_ft = TDNN(in_channels=TDNN_feats)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        #PANN models
        # elif model_name == 'CNN_14':
        #     #feature parameters from repo
        #     if use_pretrained: #Pretrained on AudioSet
        #         model_ft = Cnn14(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, 
        #             fmax=8000, classes_num=527)
        #         model_ft.load_state_dict(torch.load('./PANN_Weights/Cnn14_16k_mAP=0.438.pth')['model'])
        #     else:
        #         model_ft = Cnn14(sample_rate=16000, window_size=250, hop_size=64, mel_bins=64, fmin=0, 
        #             fmax=None, classes_num=num_classes)
                
        #     set_parameter_requires_grad(model_ft, feature_extract)
        #     num_ftrs = model_ft.fc_audioset.in_features
        #     model_ft.fc_audioset = nn.Linear(num_ftrs,num_classes)
        #     input_size = 224 #Verify


        elif model_name == 'CNN_14':
            if use_pretrained:
                # Download weights if not exist
                download_weights(weights_url, weights_path)
        
                # Load pretrained model
                model_ft = Cnn14(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, 
                    fmax=8000, classes_num=527)
                model_ft.load_state_dict(torch.load(weights_path)['model'])
            else:
                model_ft = Cnn14(sample_rate=16000, window_size=250, hop_size=64, mel_bins=64, fmin=0, 
                    fmax=None, classes_num=num_classes)
        
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc_audioset.in_features
            model_ft.fc_audioset = nn.Linear(num_ftrs, num_classes)
            input_size = 224  # Verify
        
            # Wrap the model with CustomPANN
            custom_model = CustomPANN(model_ft)
        
        else:
            raise RuntimeError('{} not implemented'.format(model_name))
        
        # Initialize feature layer
        if model_name == 'CNN_14':
            feature_layer = nn.Sequential()
        else:
            pass
        

        return custom_model, input_size, feature_layer
    

