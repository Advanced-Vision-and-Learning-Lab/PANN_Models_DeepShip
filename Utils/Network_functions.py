#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:51:00 2024

@author: amir.m
"""

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


import os
import requests

from Utils.PANN_models import Cnn14, ResNet38, MobileNetV1, Res1dNet31, Wavegram_Logmel_Cnn14  
import timm

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CustomPANN(nn.Module):
    def __init__(self, model):
        super(CustomPANN, self).__init__()
        self.fc = model.fc_audioset
        model.fc_audioset = nn.Sequential()
        self.backbone = model

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

def download_weights(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading weights from {url} to {destination}...\n")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Download complete.\n")
    else:
        print(f"Weights already exist at {destination}.\n")

def initialize_model(model_name, use_pretrained, feature_extract, num_classes, pretrained_loaded=False):
    model_params = {
        'CNN_14_8k': {
            'class': Cnn14,
            'pretrained_url': "https://zenodo.org/record/3987831/files/Cnn14_8k_mAP%3D0.357.pth?download=1",
            'sample_rate': 8000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 4000
        },
        'CNN_14_16k': {
            'class': Cnn14,
            'pretrained_url': "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 8000
        },
        'CNN_14_32k': {
            'class': Cnn14,
            'pretrained_url': "https://zenodo.org/record/3987831/files/Cnn14_32k_mAP%3D0.474.pth?download=1",
            'sample_rate': 32000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 16000
        },
        'ResNet38': {
            'class': ResNet38,
            'pretrained_url': "https://zenodo.org/record/3960586/files/ResNet38_mAP%3D0.434.pth?download=1",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 8000
        },
        'MobileNetV1': {
            'class': MobileNetV1,
            'pretrained_url': "https://zenodo.org/record/3960586/files/MobileNetV1_mAP%3D0.389.pth?download=1",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 8000
        },
        'Res1dNet31': {
            'class': Res1dNet31,
            'pretrained_url': "https://zenodo.org/record/3960586/files/Res1dNet31_mAP%3D0.365.pth?download=1",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 8000
        },
        'Wavegram-Logmel-CNN': {
            'class': Wavegram_Logmel_Cnn14,
            'pretrained_url': "https://zenodo.org/record/3960586/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 8000
            },
        'EfficientNet-B0': {
            'class': 'efficientnet_b0',
            'pretrained': True,
            'input_size': 224
        },
        'ResNet50': {
            'class': 'resnet50',
            'pretrained': True,
            'input_size': 224
        },
        'ViT-B/16': {
            'class': 'vit_base_patch16_224',
            'pretrained': True,
            'input_size': 224
        },
        'DenseNet121': {
            'class': 'densenet121',
            'pretrained': True,
            'input_size': 224
        },
        'ConvNext-Tiny': {
            'class': 'convnext_tiny',
            'pretrained': True,
            'input_size': 224
        }
    }

    if model_name not in model_params:
        raise RuntimeError('{} not implemented'.format(model_name))

    params = model_params[model_name]

    if 'pretrained_url' in params:
        # Logic for PANN models
        model_class = params['class']
        weights_url = params['pretrained_url']
        sample_rate = params['sample_rate']
        window_size = params['window_size']
        hop_size = params['hop_size']
        mel_bins = params['mel_bins']
        fmin = params['fmin']
        fmax = params['fmax']
        weights_path = f"./PANN_Weights/{model_name}_weights.pth"
        
        model_ft = model_class(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=527)
        
        if use_pretrained and not pretrained_loaded:
            if not os.path.exists(weights_path) or os.path.getsize(weights_path) == 0:
                download_weights(weights_url, weights_path)
            try:
                state_dict = torch.load(weights_path)
                model_ft.load_state_dict(state_dict['model'])
            except Exception as e:
                raise RuntimeError(f"Error loading the model weights: {e}")

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc_audioset.in_features
        model_ft.fc_audioset = nn.Linear(num_ftrs, num_classes)
        custom_model = CustomPANN(model_ft)
        feature_layer = nn.Sequential()

    else:
        # Logic for TIMM models
        model_class = params['class']
        input_size = params['input_size']
        
        if use_pretrained and not pretrained_loaded:
            model_ft = timm.create_model(model_class, pretrained=True)
        else:
            model_ft = timm.create_model(model_class, pretrained=False)

        set_parameter_requires_grad(model_ft, feature_extract)
        
        if 'fc' in dir(model_ft):
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        elif 'classifier' in dir(model_ft):
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        elif 'head' in dir(model_ft):
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Linear(num_ftrs, num_classes)

        custom_model = model_ft
        feature_layer = nn.Sequential()

    return custom_model, input_size, feature_layer

