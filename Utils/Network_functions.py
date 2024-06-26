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


def initialize_model(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=5,
                     feat_map_size=4, TDNN_feats=1, input_feature='STFT',RGB=True,
                     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    
    #If TDNN model, only use 1 feature channel
    if model_name == "TDNN":
        RGB = False
 

    #If TDNN model, only use 1 feature channel
    if model_name == "AST":
        RGB = False    

    
    #Intialize feature layer
    feature_layer= Feature_Extraction_Layer(input_feature=input_feature,
                                             RGB=RGB)
    ft_dims = feature_layer.output_dims
    #print(ft_dims)
    #pdb.set_trace()
    
    model_ft = None
    input_size = 0
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes(histogram_layer,parallel=parallel,
                           model_name=model_name,add_bn=add_bn,scale=scale,
                           pretrained=use_pretrained, TDNN_feats=TDNN_feats)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)
        
        

        #Reduce number of conv channels from input channels to input 
        #channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels/feat_map_size)/(histogram_layer.numBins)) 
                                                                                   
        print('reduced_dim: ', reduced_dim)
        #pdb.set_trace()   
        
        
        
        
        # #If input channels equals reduced/increase, don't apply 1x1 convolution
        if (in_channels==reduced_dim): 
            model_ft.histogram_layer = histogram_layer
        else:
            #conv_reduce = nn.Conv2d(in_channels,reduced_dim,(1,1))
            conv_reduce = nn.Conv1d(in_channels,reduced_dim,1) 
            model_ft.histogram_layer = nn.Sequential(conv_reduce,histogram_layer)
            

            
            
            
        
        # if(parallel):
        #     num_ftrs = model_ft.fc.in_features*2             
        # else:
        #     num_ftrs = model_ft.fc.in_features
        
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

      
        #AST
        if parallel:
            num_ftrs = 1536 
            #num_ftrs = 1152
        else:
            num_ftrs = 768  




        add_ln=False
        
        # model_ft.fc = nn.Sequential(
        #     nn.LayerNorm(num_ftrs) if add_ln else nn.Identity(),
        #     nn.Linear(num_ftrs, num_classes)
        # )        
        
        
        #pdb.set_trace()   
        feat_ex = False
        if feat_ex:
            for param in model_ft.parameters():
                param.requires_grad = False
            
            # Unfreeze parameters in the mlp_head
            for param in model_ft.fc.parameters():
                param.requires_grad = True
                   
                
  
        input_size = 224

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
        elif model_name == 'CNN_14':
            #feature parameters from repo
            if use_pretrained: #Pretrained on AudioSet
                model_ft = Cnn14(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, 
                    fmax=8000, classes_num=527)
                model_ft.load_state_dict(torch.load('./PANN_Weights/Cnn14_16k_mAP=0.438.pth')['model'])
            else:
                model_ft = Cnn14(sample_rate=16000, window_size=250, hop_size=64, mel_bins=64, fmin=0, 
                    fmax=None, classes_num=num_classes)
                
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc_audioset.in_features
            model_ft.fc_audioset = nn.Linear(num_ftrs,num_classes)
            input_size = 224 #Verify


        elif model_name == "AST":
            
            
            
            inpf = ft_dims[1]
            inpt = ft_dims[2]
            

            t_mode = 'linear_probing'
            if t_mode == 'adapters':
                Use_A = True
            else:
                Use_A = False
            
            a_location = 'ffn'  # Can be 'ffn', 'mhsa', or 'both'
            a_mode = 'parallel' # 'sequential', 'parallel'
            
            h_mode = 'histogram'
            if h_mode == 'histogram':
                Use_H = True
                h_operation = 'add'  # options: 'concat', 'add', 'multiply'
                h_location = 'within_each'  # options: 'within_each', 'after_encoder'
   
            else:
                Use_H = False
                h_operation = None
                h_location = None
                histogram_layer = None
            
            model_ft = ASTModel(label_dim=4, input_fdim=inpf, input_tdim=inpt, 
                                imagenet_pretrain=True, audioset_pretrain=True, 
                                use_adapters=Use_A, adapter_mode=a_mode, adapter_location=a_location,
                                use_histogram=Use_H, histogram_layer=histogram_layer, histogram_operation=h_operation, histogram_location=h_location)
            
            if t_mode == 'full_fine_tune':
                for param in model_ft.parameters():
                    param.requires_grad = True
            elif t_mode == 'linear_probing':
                for param in model_ft.parameters():
                    param.requires_grad = False
                for param in model_ft.mlp_head.parameters():
                    param.requires_grad = True
            elif t_mode == 'adapters':
                for param in model_ft.parameters():
                    param.requires_grad = False
                if a_location in ['both', 'mhsa']:
                    for param in model_ft.adapters_mhsa.parameters():
                        param.requires_grad = True
                if a_location in ['both', 'ffn']:
                    for param in model_ft.adapters_ffn.parameters():
                        param.requires_grad = True
                for param in model_ft.mlp_head.parameters():
                    param.requires_grad = True
            
            if Use_H and h_location == 'after_encoder':
                for param in model_ft.histogram_layer.parameters():
                    param.requires_grad = True
                    
            if Use_H and h_location == 'within_each':
                for param in model_ft.histogram_layers.parameters():
                    param.requires_grad = True        

                #set_parameter_requires_grad(model_ft, feature_extract)
    
    
    
                input_size = 224  
            
         
        else:
            raise RuntimeError('{} not implemented'.format(model_name))

    #Intialize feature layer
    if model_name == 'CNN_14':
        feature_layer = nn.Sequential()
    else:
        pass
    
    #Return baseline model, desired input size, and feature layer
    return model_ft, input_size, feature_layer, ft_dims


