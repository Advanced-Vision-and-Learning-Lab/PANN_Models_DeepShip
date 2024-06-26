# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:15:42 2023
Code modified from: https://github.com/lucascesarfd/underwater_snd/blob/master/nauta/one_stage/dataset.py
@author: jpeeples
"""
import pdb
import torch
import os
from scipy.io import wavfile
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torchaudio

# class DeepShipSegments(Dataset):
#     def __init__(self, parent_folder, train_split=.7,val_test_split=.5,
#                  partition='train', random_seed= 42, shuffle = False, transform=None, 
#                  target_transform=None):
#         self.parent_folder = parent_folder
#         self.folder_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }
#         self.train_split = train_split
#         self.val_test_split = val_test_split
#         self.partition = partition
#         self.transform = transform
#         self.shuffle = shuffle
#         self.target_transform = target_transform
#         self.random_seed = random_seed
#         self.norm_function = None
#         self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}

#         # Loop over each label and subfolder
#         for label in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
#             label_path = os.path.join(parent_folder, label)
#             subfolders = os.listdir(label_path)
            
#             # Split subfolders into training, testing, and validation sets
#             subfolders_train, subfolders_test_val = train_test_split(subfolders, 
#                                                                      train_size=train_split, 
#                                                                      shuffle=self.shuffle, 
#                                                                      random_state=self.random_seed)
#             subfolders_test, subfolders_val = train_test_split(subfolders_test_val, 
#                                                                train_size=self.val_test_split, 
#                                                                shuffle=self.shuffle, 
#                                                                random_state=self.random_seed)

#             # Add subfolders to appropriate folder list
#             for subfolder in subfolders_train:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['train'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_test:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['test'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_val:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['val'].append((subfolder_path, self.class_mapping[label]))

#         self.segment_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }

#         # Loop over each folder list and add corresponding files to file list
#         for split in ['train', 'test', 'val']:
#             for folder in self.folder_lists[split]:
#                 for root, dirs, files in os.walk(folder[0]):
#                     for file in files:
#                         if file.endswith('.wav'):
#                             file_path = os.path.join(root, file)
#                             label = folder[1]
#                             self.segment_lists[split].append((file_path, label))

#     def __len__(self):
#         return len(self.segment_lists[self.partition])

#     def __getitem__(self, idx):
#         file_path, label = self.segment_lists[self.partition][idx]    
        
        
        
#         sr, signal = wavfile.read(file_path, mmap=False)
#         signal = signal.astype(np.float32)
        

#         # Perform min-max normalization
#         if self.norm_function is not None:
#             signal = self.norm_function(signal)
#             signal = torch.tensor(signal)

        
#         label = torch.tensor(label)
#         if self.target_transform:
#             label = self.target_transform(label)

#         return signal, label, idx
    
    
    
    
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
import torch
from scipy.io import wavfile
import pandas as pd
from sklearn.model_selection import StratifiedKFold


from collections import defaultdict

class DeepShipSegments(torch.utils.data.Dataset):
    def __init__(self, data_dir, run_number):
        self.data_dir = data_dir
        self.run_number = run_number
        self.subfolder_map = {}  
        self.data, self.labels, self.subfolder_labels = self._load_data()
        self.min_val, self.max_val = None, None

    def _load_data(self):
        data = []
        labels = []
        subfolder_labels = []

        subfolder_counter = 0

        for main_label, main_folder in enumerate(os.listdir(self.data_dir)):
            main_path = os.path.join(self.data_dir, main_folder)
            if os.path.isdir(main_path):
                for subfolder in os.listdir(main_path):
                    subfolder_path = os.path.join(main_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        if subfolder_path not in self.subfolder_map:
                            self.subfolder_map[subfolder_path] = subfolder_counter
                            subfolder_counter += 1
                        segments = [os.path.join(subfolder_path, wav) for wav in os.listdir(subfolder_path) if wav.endswith('.wav')]
                        data.extend(segments)
                        labels.extend([main_label] * len(segments))
                        subfolder_labels.extend([self.subfolder_map[subfolder_path]] * len(segments))

        return data, labels, subfolder_labels


    def get_train_val_indices(self, fold_index, num_folds):
        # Create a list of unique subfolder indices
        unique_subfolders = list(set(self.subfolder_labels))
        
        # Sort to ensure consistent ordering 
        unique_subfolders.sort()
        
        # Calculate the number of subfolders per fold
        num_subfolders_per_fold = len(unique_subfolders) // num_folds
        
        # Split subfolders into folds
        folds = [unique_subfolders[i * num_subfolders_per_fold:(i + 1) * num_subfolders_per_fold] for i in range(num_folds)]
        
        # Allow for the last fold to take any remaining subfolders due to integer division
        if len(unique_subfolders) % num_folds != 0:
            folds[-1].extend(unique_subfolders[num_folds * num_subfolders_per_fold:])
    
        # Validation subfolders for this fold
        val_subfolders = folds[fold_index]
        
        # Training subfolders are all other subfolders
        train_subfolders = [sf for i, fold in enumerate(folds) if i != fold_index for sf in fold]
        
        # Convert subfolder indices back to segment indices
        train_indices = [i for i, sf in enumerate(self.subfolder_labels) if sf in train_subfolders]
        val_indices = [i for i, sf in enumerate(self.subfolder_labels) if sf in val_subfolders]
    
        return train_indices, val_indices

    def set_normalization(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_path = self.data[index]
        label = self.labels[index]
        sample_rate, audio_data = wavfile.read(wav_path)
        audio_data = audio_data.astype(np.float32)
        if self.min_val is not None and self.max_val is not None:
            audio_data = (audio_data - self.min_val) / (self.max_val - self.min_val)
        return audio_data, label


    def summarize_data_usage(self, train_indices, val_indices, fold_index):
        train_folders = {}
        val_folders = {}
        
        # Reverse mapping to get folder path from subfolder index
        index_to_subfolder = {v: k for k, v in self.subfolder_map.items()}
        
        # Count segments per subfolder
        for idx in train_indices:
            subfolder_index = self.subfolder_labels[idx]
            folder_path = index_to_subfolder[subfolder_index]
            label = self.labels[idx]
            if folder_path not in train_folders:
                train_folders[folder_path] = {'count': 0, 'label': label}
            train_folders[folder_path]['count'] += 1
        
        for idx in val_indices:
            subfolder_index = self.subfolder_labels[idx]
            folder_path = index_to_subfolder[subfolder_index]
            label = self.labels[idx]
            if folder_path not in val_folders:
                val_folders[folder_path] = {'count': 0, 'label': label}
            val_folders[folder_path]['count'] += 1
        
        # Write summaries
        summary_path = os.path.join(self.data_dir, f'summary_run_{self.run_number}_fold_{fold_index}.txt')
        with open(summary_path, 'w') as file:
            file.write("Training Folders and Segment Counts:\n")
            for folder, info in train_folders.items():
                file.write(f"{folder}: {info['count']} segments\n")
        
            file.write("\nValidation Folders and Segment Counts:\n")
            for folder, info in val_folders.items():
                file.write(f"{folder}: {info['count']} segments\n")
        
            # Overall totals
            total_train_segments = sum(info['count'] for info in train_folders.values())
            total_val_segments = sum(info['count'] for info in val_folders.values())
            file.write(f"\nTotal Training Segments: {total_train_segments}\n")
            file.write(f"Total Validation Segments: {total_val_segments}\n")





    
    
        
