#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:51:39 2024

@author: amir.m
"""
from __future__ import print_function
from __future__ import division
from Demo_Parameters import Parameters
import numpy as np
import argparse
import random
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt

from Datasets.Get_preprocessed_data import process_data
import numpy as np
# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from SSDataModule import SSAudioDataModule

import librosa.display
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# Define the MelSpectrogramExtractor class
class MelSpectrogramExtractor(nn.Module): 
    def __init__(self, sample_rate=32000, n_fft=1024, win_length=1024, hop_length=320, n_mels=64, fmin=50, fmax=14000):
        super(MelSpectrogramExtractor, self).__init__()
        
        # Settings for Spectrogram
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                                  win_length=win_length, window=window, center=center, 
                                                  pad_mode=pad_mode, 
                                                  power=2.0, freeze_parameters=True)

        # Logmel feature extractor
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=win_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        
    def forward(self, waveform):
        waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        spectrogram = self.spectrogram_extractor(waveform)
        log_mel_spectrogram = self.logmel_extractor(spectrogram)
        return log_mel_spectrogram

    def save_spectrogram_figure(self, spectrogram, filename='spectrogram.png', dpi=500):
        spectrogram = spectrogram.squeeze().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB').set_label('Power')
        plt.title('Log Mel Spectrogram')
        plt.xlabel('Mel Bins')
        plt.ylabel('Time Frames')
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()
        print(f'Saved spectrogram figure with shape: {spectrogram.shape}')


def main(Params):
    # Parameters setup
    Dataset = Params['Dataset']
    model_name = Params['Model_name']
    num_classes = Params['num_classes'][Dataset]
    batch_size = Params['batch_size']['train']
    sample_rate = Params['sample_rate']

    print('\nStarting Experiments...')

    # Data module setup
    new_dir = Params["new_dir"]
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=sample_rate)
    data_module.prepare_data()

    # Retrieve raw and normalized audio data
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    sample_normalized, _ = batch  # Normalized sample
    raw_audio = data_module.get_raw_audio_data()  # Retrieve raw audio data

    # Convert normalized audio tensor to NumPy array for processing
    audio_tensor_normalized = sample_normalized[0].numpy() if isinstance(sample_normalized[0], torch.Tensor) else sample_normalized[0]
    
    # Instantiate the MelSpectrogramExtractor with appropriate parameters
    mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
    
    # Convert the audio_tensor_normalized to torch tensor and compute spectrogram
    audio_tensor_normalized_torch = torch.from_numpy(audio_tensor_normalized).float()
    spectrogram = mel_extractor(audio_tensor_normalized_torch)
    
    # Save the spectrogram
    mel_extractor.save_spectrogram_figure(spectrogram, 'features/normalized_audio_spectrogram.png', dpi=500)

    # Plot the raw audio waveform
    plt.figure(figsize=(6, 4))
    plt.plot(raw_audio)  # Plotting the raw audio data directly
    plt.title('Raw Audio Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('features/raw_audio_waveform_samples.png', dpi=500)

    # Plot the normalized audio waveform
    plt.figure(figsize=(6, 4))
    plt.plot(audio_tensor_normalized)  # Plotting the normalized audio data
    plt.title('Normalized Audio Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Normalized Amplitude')
    plt.tight_layout()
    plt.savefig('features/normalized_audio_waveform_samples.png', dpi=500)

    # Print shapes for documentation in your paper
    print("Shape of the raw audio data:", raw_audio.shape)
    print("Shape of the normalized audio data:", audio_tensor_normalized.shape)





def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='densenet201', #CNN_14_16k #CNN_14_16k #ViT-B/16
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Select optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Dataset Sample Rate')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
