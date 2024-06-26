import torch
import torch.nn as nn
import torchaudio
import torch.nn as nn
from nnAudio import features
import torchaudio 
import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio

class FBankLayer(nn.Module):
    def __init__(self, sample_frequency=16000.0, num_mel_bins=40, frame_length=25.0, 
                 frame_shift=10.0, window_type='hann', use_log_fbank=True, use_power=True):
        super(FBankLayer, self).__init__()
        self.sample_frequency = sample_frequency
        self.num_mel_bins = num_mel_bins
        self.frame_length = int(sample_frequency * frame_length / 1000)  # Convert from ms to samples
        self.frame_shift = int(sample_frequency * frame_shift / 1000)    # Convert from ms to samples
        self.window_type = window_type
        self.use_log_fbank = use_log_fbank
        self.use_power = use_power

        self.sample_frequency = sample_frequency 
        window_length = frame_length
        hop_length = frame_shift
        
        # Create the window function
        if window_type == 'hann':
            self.window = torch.hann_window(self.frame_length)
        elif window_type == 'hamming':
            self.window = torch.hamming_window(self.frame_length)
        elif window_type == 'blackman':
            self.window = torch.blackman_window(self.frame_length)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        self.stft = nn.Sequential(features.STFT(sr=sample_frequency,n_fft=int(window_length*sample_frequency), 
                                        hop_length=int(hop_length*sample_frequency),
                                        win_length=int(window_length*sample_frequency), trainable=False,
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        # Create the Mel filter bank
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=num_mel_bins,
            sample_rate=sample_frequency,
            n_stft=self.frame_length // 2 + 1,
            f_min=0.0,
            f_max=sample_frequency / 2
        )

    def forward(self, waveforms):
        # Move window to the same device as waveforms
        window = self.window.to(waveforms.device)
        
        # Compute the STFT
        #stft = torch.stft(waveforms, n_fft=self.frame_length, hop_length=self.frame_shift, 
        #                  win_length=self.frame_length, window=window, return_complex=True)
        
        stft = self.stft(waveforms)
        
        # Compute the magnitude or power of the STFT
        if self.use_power:
            spectrogram = stft.abs() ** 2
        else:
            spectrogram = stft.abs()

        # Apply the Mel filter bank
        mel_spectrogram = self.mel_scale(spectrogram)

        if self.use_log_fbank:
            mel_spectrogram = torch.log(mel_spectrogram + 1e-6)  # Add a small value to avoid log(0)

        return mel_spectrogram

