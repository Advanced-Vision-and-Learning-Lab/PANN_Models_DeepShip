#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:51:00 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division
import numpy as np

## PyTorch dependencies
import torch
import torch.nn as nn

import pdb

import os
import requests

from Utils.PANN_models import Cnn14, ResNet38, MobileNetV1, Res1dNet31, Wavegram_Logmel_Cnn14  
import timm

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class MelSpectrogramExtractor(nn.Module): 
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=160, n_mels=64, fmin=50, fmax=8000):
        super(MelSpectrogramExtractor, self).__init__()
        
        # Settings for Spectrogram
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        
        self.spectrogram_extractor = Spectrogram(n_fft=win_length, hop_length=hop_length, 
                                                 win_length=win_length, window=window, center=center, 
                                                 pad_mode=pad_mode, 
                                                 freeze_parameters=True)

        # Logmel feature extractor
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=win_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
    def forward(self, waveform):
        spectrogram = self.spectrogram_extractor(waveform)
        log_mel_spectrogram = self.logmel_extractor(spectrogram)
        augmented_log_mel_spectrogram = self.spec_augmenter(log_mel_spectrogram)
        return augmented_log_mel_spectrogram


class CustomPANN(nn.Module):
    def __init__(self, model):
        super(CustomPANN, self).__init__()
        
        self.fc = model.fc_audioset
        model.fc_audioset = nn.Sequential()
        
        self.backbone = model

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.fc(features)
        return features, predictions
    
    
class CustomTIMM(nn.Module):
    def __init__(self, model):
        super(CustomTIMM, self).__init__()
        #self.fc = None
        if 'fc' in dir(model):
            self.fc = model.fc
            model.fc = nn.Sequential()
        elif 'classifier' in dir(model):
            self.fc = model.classifier
            model.classifier = nn.Sequential()
        elif 'head' in dir(model):
            self.fc = model.head
            model.head = nn.Sequential()
        
        self.backbone = model

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.fc(features)
        return features, predictions

    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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
            'pretrained_url': "https://zenodo.org/records/3987831/files/Cnn14_8k_mAP%3D0.416.pth?download=1",
            'weights_name': "Cnn14_8k_mAP=0.416.pth",
            'sample_rate': 8000, 'window_size': 256, 'hop_size': 80, 'mel_bins': 64, 'fmin': 50, 'fmax': 3500
        },
        'CNN_14_16k': {
            'class': Cnn14,
            'pretrained_url': "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",   
            'weights_name': "Cnn14_16k_mAP=0.438.pth",
            'sample_rate': 16000, 'window_size': 512, 'hop_size': 160, 'mel_bins': 64, 'fmin': 50, 'fmax': 7000
        },
        'CNN_14_32k': {
            'class': Cnn14,
            'pretrained_url': "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
            'weights_name': "Cnn14_mAP=0.431.pth",
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'ResNet38': {
            'class': ResNet38,
            'pretrained_url': "https://zenodo.org/record/3960586/files/ResNet38_mAP%3D0.434.pth?download=1",
            'weights_name': "ResNet38_mAP=0.434.pth",
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'MobileNetV1': {
            'class': MobileNetV1,
            'pretrained_url': "https://zenodo.org/record/3960586/files/MobileNetV1_mAP%3D0.389.pth?download=1",
            'weights_name': "MobileNetV1_mAP=0.389.pth",
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'Res1dNet31': {
            'class': Res1dNet31,
            'pretrained_url': "https://zenodo.org/record/3960586/files/Res1dNet31_mAP%3D0.365.pth?download=1",
            'weights_name': "Res1dNet31_mAP=0.365.pth",
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'Wavegram_Logmel_Cnn14': {
            'class': Wavegram_Logmel_Cnn14,
            'pretrained_url': "https://zenodo.org/record/3960586/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
            'weights_name': "Wavegram_Logmel_Cnn14_mAP=0.439.pth",
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'efficientnet_b3': {
            'class': 'efficientnet_b3',
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'resnet50': {
            'class': 'resnet50',
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'densenet121': {
            'class': 'densenet121',
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        },
        'mobilenetv3_large_100': {
            'class': 'mobilenetv3_large_100', 
            'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000
        }
    }

    if model_name not in model_params:
        raise RuntimeError('{} not implemented'.format(model_name))

    params = model_params[model_name]
    
    if 'pretrained_url' in params:
          # PANN models
          model_class = params['class']
          weights_url = params['pretrained_url']
          sample_rate = params['sample_rate']
          window_size = params['window_size']
          hop_size = params['hop_size']
          mel_bins = params['mel_bins']
          fmin = params['fmin']
          fmax = params['fmax']
    
          weights_name = params['weights_name']  
          weights_path = f"./PANN_Weights/{weights_name}"  
    
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
          mel_extractor = nn.Sequential() 

    
    else:
          # TIMM models
          model_class = params['class']
    
          if use_pretrained and not pretrained_loaded:
              model_ft = timm.create_model(model_class, pretrained=True, in_chans=1)
          else:
              model_ft = timm.create_model(model_class, pretrained=False, in_chans=1)
    
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
        
          custom_model = CustomTIMM(model_ft)
          
          mel_extractor = MelSpectrogramExtractor(sample_rate=params['sample_rate'], 
                                                    win_length=params['window_size'], 
                                                    hop_length=params['hop_size'], 
                                                    n_mels=params['mel_bins'], 
                                                    fmin=params['fmin'], 
                                                    fmax=params['fmax'])



    return custom_model, mel_extractor
          
          
