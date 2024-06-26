import torch.nn as nn
from nnAudio import features
import torchaudio 
import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np
from .FBankLayer import FBankLayer  # Adjust the import path if necessary

class Fbank(nn.Module):
    def __init__(self, num_mel_bins=128, sample_frequency=16000, frame_shift=10, 
                  frame_length=25):
        super(Fbank, self).__init__()
        
        
        self.num_mel_bins = num_mel_bins
        self.sample_frequency = sample_frequency
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        
        self.spec_norm_f = False
        self.dataset_mean_f = 11.575 #Fbank
        self.dataset_std_f = 3.295

            
    def forward(self, x):
       
        x_spec = []
        for sample in range(x.shape[0]):
            x_spec.append(torchaudio.compliance.kaldi.fbank(x[sample].unsqueeze(0), num_mel_bins=self.num_mel_bins, 
                                                sample_frequency=self.sample_frequency, 
                                                frame_shift=self.frame_shift, 
                                                frame_length=self.frame_length).unsqueeze(0))
        
        x_spec = torch.cat(x_spec,dim=0)
        x = x_spec.transpose(1, 2)

        #torch.Size([1, 128, 498])

        
        if self.spec_norm_f:
            x = (x - self.dataset_mean_f) / (self.dataset_std_f * 2)
        
        return x



class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=250, 
                 hop_length=64, RGB=False, freq_m = 24, time_m = 96, frame_shift=10.0):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate  
        self.freq_m = freq_m
        self.time_m = time_m
        
        self.sample_frequency = sample_rate 
        self.frame_shift = frame_shift  

        
        self.spec_norm = False
        self.dataset_mean = -0.496
        self.dataset_std = 2.16
        
        
        #Define training augmentations
        self.freqm = torchaudio.transforms.FrequencyMasking(self.freq_m)
        self.timem = torchaudio.transforms.TimeMasking(self.time_m)
        #hop_length = 16

     
        window_length = 250
        hop_length = 64
        #window_length = 25
        #hop_length = 10
        
        #Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        
        
        window_length_H = 100
        hop_length_H = 25
        window_length_H /= 1000
        hop_length_H /= 1000
        
        
        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3,2,16,16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1,4,0,0))
        
        self.num_channels = num_channels
        self.input_feature = input_feature

        #Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), nn.ZeroPad2d((1,0,8,0)))
                                            

    
    
        #Return MFCC that is 16 x 48 (TDNN models) or 48 x 48 (CNNs)
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), MFCC_padding)

        #Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), trainable=False,
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return STFT that is 128 x 202
        self.STFT_H = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length_H*sample_rate), 
                                        hop_length=int(hop_length_H*sample_rate),
                                        win_length=int(window_length_H*sample_rate), 
                                        output_format='Magnitude',trainable=False,
                                        freq_bins=64,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                hop_length=int(hop_length*sample_rate),
                                                n_fft=int(window_length*sample_rate),
                                                verbose=False,n_bins=64), nn.ZeroPad2d((1,0,0,0)))
        

        #Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        

        
        # Initialize original Fbank
        self.Fbank = Fbank(num_mel_bins=128, sample_frequency=sample_rate, frame_shift=frame_shift, 
                     frame_length=25)

        # Initialize new FBankLayer
        self.FBankLayer = FBankLayer(sample_frequency=sample_rate, num_mel_bins=64, 
                                     frame_length=25.0, frame_shift=frame_shift, 
                                     window_type='hann', use_log_fbank=True, use_power=True)

        self.features = {'Mel_Spectrogram': self.Mel_Spectrogram, 
                         'MFCC': self.MFCC, 'STFT': self.STFT, 'STFT_H': self.STFT_H, 'GFCC': self.GFCC, 
                         'CQT': self.CQT, 'VQT': self.VQT, 'Fbank': self.Fbank, 'FBankLayer': self.FBankLayer}
        
        
        
        
        self.output_dims = None
        self.calculate_output_dims()


    
    
    def calculate_output_dims(self):
        try:
            length_in_seconds = 5  # Duration that ensures a reasonable number of frames
            samples = int(self.sample_rate * length_in_seconds)
            dummy_input = torch.randn(1, samples)  
            with torch.no_grad():
                output = self.features[self.input_feature](dummy_input)
                self.output_dims = output.shape
        except Exception as e:
            print(f"Failed to calculate output dimensions: {e}")
            self.output_dims = None
            

        
    def forward(self, x):
       
        #Extract audio feature
        x = self.features[self.input_feature](x).unsqueeze(1)
        
        

        # Extract features using original Fbank
        # fbank_output = self.features['Fbank'](x).unsqueeze(1)
        
        # # Extract features using new FBankLayer
        # fbank_layer_output = self.features['FBankLayer'](x).unsqueeze(1)
        
        # self.visualize_features(fbank_output, 'Original Fbank', 'Original_Fbank.png', is_log_scale=True)

        # self.visualize_features(fbank_layer_output, 'New FBankLayer', 'New_FBankLayer.png', is_log_scale=True)
        
        # # Use pdb to set a breakpoint for debugging
        # pdb.set_trace()

        
        #If training, apply augmentation
        # if self.train:
        #     x = self.freqm(x)
        #     x = self.timem(x)
    
        #torch.Size([128, 1, 128, 202])
    
        if self.spec_norm:
            x = (x - self.dataset_mean) / (self.dataset_std * 2)
        
        #Repeat channel dimension if needed (CNNs)
        x = x.repeat(1, self.num_channels,1,1)

        return x

        
        
    
    def visualize_features(self, feature_output, feature_name, save_path, is_log_scale=True):

        # Extract the shape for clearer code
        num_mel_bins = feature_output.shape[2]
        num_time_frames = feature_output.shape[3]
    
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(feature_output[0][0].cpu().detach().numpy(), aspect='auto', origin='lower')
        ax.set_title(f'{feature_name}')
        ax.set_xlabel('Time Bins')
        ax.set_ylabel('Mel Bins')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Log-Magnitude (dB)' if is_log_scale else 'Linear Magnitude')

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)  
        plt.close() 

