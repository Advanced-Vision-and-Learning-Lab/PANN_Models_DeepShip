#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:22:15 2024

@author: amir.m
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from scipy.io import wavfile
import lightning as L

class AudioDataset(Dataset):
    def __init__(self, data_list, class_to_idx):
        self.data_list = data_list  # Store the list of data
        self.class_to_idx = class_to_idx  # Store the class-to-index mapping

    def __len__(self):
        return len(self.data_list)  # Return the number of samples

    def __getitem__(self, idx):
        file_data = self.data_list[idx]  # Get the data for the given index
        data = file_data['data']  # Extract the normalized audio data
        class_name = file_data['file_path'].split(os.sep)[-3]  # Extract the class name from the file path
        label = self.class_to_idx[class_name]  # Convert the class name to an integer label
        data_tensor = torch.tensor(data, dtype=torch.float32)  # Convert data to a PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.long)  # Convert label to a PyTorch tensor
        return data_tensor, label_tensor  # Return the data and label tensors

class AudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=256, num_folds=3):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.class_to_idx = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self.train_folds = []
        self.val_folds = []
        self.prepared = False  
        self.train_indices = []
        self.val_indices = []
        self.all_recording_names = []  
        
    def list_wav_files(self):
        wav_files = []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for recording in os.listdir(class_path):
                    recording_path = os.path.join(class_path, recording)
                    if os.path.isdir(recording_path):
                        for segment in os.listdir(recording_path):
                            if segment.endswith('.wav'):
                                segment_path = os.path.join(recording_path, segment)
                                wav_files.append(segment_path)
        print(f'Found {len(wav_files)} .wav files')
        return wav_files

    def read_wav_files(self, wav_files):
        data_list = []
        for file_path in wav_files:
            sampling_rate, data = wavfile.read(file_path)
            file_data = {
                'file_path': file_path,
                'sampling_rate': sampling_rate,
                'data': data
            }
            data_list.append(file_data)
        print(f'Read {len(data_list)} .wav files')
        return data_list

    def organize_data(self, data_list):
        organized_data = defaultdict(lambda: defaultdict(list))
        for file_data in data_list:
            path_parts = file_data['file_path'].split(os.sep)
            class_name = path_parts[-3]
            recording_name = path_parts[-2]
            organized_data[class_name][recording_name].append(file_data)
        print(f'Organized data into {len(organized_data)} classes')
        return organized_data


    # def create_stratified_k_folds(self, organized_data):
    #     all_recording_names = []
    #     class_labels = []
    #     train_folds = []
    #     val_folds = []
    #     train_indices = []
    #     val_indices = []

    #     for class_name, recordings in organized_data.items():
    #         for recording_name in recordings.keys():
    #             all_recording_names.append((class_name, recording_name))
    #             class_labels.append(class_name)

    #     skf = StratifiedKFold(n_splits=self.num_folds)

    #     for train_index, val_index in skf.split(all_recording_names, class_labels):
    #         train_data = []
    #         val_data = []

    #         for idx in train_index:
    #             class_name, recording_name = all_recording_names[idx]
    #             train_data.extend(organized_data[class_name][recording_name])

    #         for idx in val_index:
    #             class_name, recording_name = all_recording_names[idx]
    #             val_data.extend(organized_data[class_name][recording_name])

    #         train_folds.append(train_data)
    #         val_folds.append(val_data)
    #         train_indices.append(train_index)
    #         val_indices.append(val_index)
            
    #     self.all_recording_names = all_recording_names 
    #     print(f'Created {len(train_folds)} training folds and {len(val_folds)} validation folds')
    #     return train_folds, val_folds, train_indices, val_indices
    
    def create_stratified_k_folds(self, organized_data):
        all_recording_names = []  # List to hold all recording names and their classes
        class_labels = []  # List to hold corresponding class labels
        train_folds = []  # List to hold training data for each fold
        val_folds = []  # List to hold validation data for each fold
        train_indices = []  # List to hold training indices for each fold
        val_indices = []  # List to hold validation indices for each fold
    
        # Collect all recording names and their classes
        for class_name, recordings in organized_data.items():
            for recording_name in recordings.keys():
                all_recording_names.append((class_name, recording_name))
                class_labels.append(class_name)
    
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=self.num_folds)
    
        # Perform the stratified k-fold split
        for train_index, val_index in skf.split(all_recording_names, class_labels):
            train_data = []  # List to hold training data for the current fold
            val_data = []  # List to hold validation data for the current fold
    
            # Collect training data based on the indices
            for idx in train_index:
                class_name, recording_name = all_recording_names[idx]
                train_data.extend(organized_data[class_name][recording_name])
    
            # Collect validation data based on the indices
            for idx in val_index:
                class_name, recording_name = all_recording_names[idx]
                val_data.extend(organized_data[class_name][recording_name])
    
            # Append the data and indices for the current fold
            train_folds.append(train_data)
            val_folds.append(val_data)
            train_indices.append(train_index)
            val_indices.append(val_index)
    
        # Store all recording names for later use
        self.all_recording_names = all_recording_names
        print(f'Created {len(train_folds)} training folds and {len(val_folds)} validation folds')
    
        # Return the training folds, validation folds, and their corresponding indices
        return train_folds, val_folds, train_indices, val_indices
    
    
    

    def check_data_leakage(self, train_folds, val_folds):
        print("Checking data leakage")
        sample_counts = defaultdict(lambda: {'train': 0, 'val': 0})
        for i in range(self.num_folds):
            train_data = train_folds[i]
            val_data = val_folds[i]

            train_set = set(file_data['file_path'] for file_data in train_data)
            val_set = set(file_data['file_path'] for file_data in val_data)
            overlap = train_set.intersection(val_set)

            if overlap:
                print(f"\nData leakage detected in fold {i + 1}! Overlap samples: {len(overlap)}\n")

            for file_data in train_data:
                sample_counts[file_data['file_path']]['train'] += 1
            for file_data in val_data:
                sample_counts[file_data['file_path']]['val'] += 1

        for file_path, counts in sample_counts.items():
            if counts['val'] != 1 or counts['train'] != self.num_folds - 1:
                print(f"Sample {file_path} does not meet the expected distribution!") 
                print(f"Train count: {counts['train']}, Val count: {counts['val']}")
    
    def count_samples_per_class(self, data_list):
        class_counts = defaultdict(int) 
        for file_data in data_list:
            class_name = file_data['file_path'].split(os.sep)[-3] 
            class_counts[class_name] += 1  
        return class_counts

    def print_class_distribution(self):
        for i in range(self.num_folds):
            train_data = self.train_folds[i]
            val_data = self.val_folds[i]
    
            train_class_counts = self.count_samples_per_class(train_data)
            val_class_counts = self.count_samples_per_class(val_data)
    
            print(f'\nFold {i}:')
            print('Training set class distribution:')
            for class_name, count in train_class_counts.items():
                print(f'  {class_name}: {count}')
    
            print('Validation set class distribution:')
            for class_name, count in val_class_counts.items():
                print(f'  {class_name}: {count}')
    

    def get_min_max_train(self, train_data):
        global_min = float('inf')
        global_max = float('-inf')
        for file_data in train_data:
            data = file_data['data'].astype(np.float32)
            file_min = np.min(data)
            file_max = np.max(data)
            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max
        return global_min, global_max

    def normalize_data(self, data_list, global_min, global_max):
        print("Normalizing train/val")
        normalized_data_list = []
        global_min = np.float32(global_min)
        global_max = np.float32(global_max)
        for file_data in data_list:
            data = file_data['data'].astype(np.float32)
            normalized_data = (data - global_min) / (global_max - global_min)
            normalized_file_data = {
                'file_path': file_data['file_path'],
                'sampling_rate': file_data['sampling_rate'],
                'data': normalized_data
            }
            normalized_data_list.append(normalized_file_data)
        return normalized_data_list
   

    def prepare_data(self):
        if not self.prepared:
            self.wav_files = self.list_wav_files()
            self.data_list = self.read_wav_files(self.wav_files)
            self.organized_data = self.organize_data(self.data_list)
            self.train_folds, self.val_folds, self.train_indices, self.val_indices = self.create_stratified_k_folds(self.organized_data)
            self.check_data_leakage(self.train_folds, self.val_folds)
            self.print_class_distribution()
            self.prepared = True
            
    def set_fold_index(self, fold_index):
        self.fold_index = fold_index
        print(f"Set fold {self.fold_index}\n")
        self._prepare_fold_specific_data()
        
        
    def _prepare_fold_specific_data(self):
        train_data = self.train_folds[self.fold_index]
        self.global_min, self.global_max = self.get_min_max_train(train_data)
        self.train_data = self.normalize_data(train_data, self.global_min, self.global_max)
        self.val_data = self.normalize_data(self.val_folds[self.fold_index], self.global_min, self.global_max)
        
        train_class_counts = self.count_samples_per_class(self.train_data)
        val_class_counts = self.count_samples_per_class(self.val_data)
    
        print(f'\nFold {self.fold_index}:')
        print('Training set class distribution:')
        for class_name, count in train_class_counts.items():
            print(f'  {class_name}: {count}')
        print('Validation set class distribution:')
        for class_name, count in val_class_counts.items():
            print(f'  {class_name}: {count}')
            
        print(f'\nGlobal min and max: {self.global_min}, {self.global_max}\n')
        
    def save_fold_indices(self, filepath):
        print("\nSaving fold split indices")
        with open(filepath, 'w') as f:
            for i in range(self.num_folds):
                f.write(f'Fold {i + 1}\n')
                f.write('Training indices and paths:\n')
                for idx in self.train_indices[i]:
                    class_name, recording_name = self.all_recording_names[idx]
                    file_paths = [file_data['file_path'] for file_data in self.organized_data[class_name][recording_name]]
                    for file_path in file_paths:
                        f.write(f'{idx}: {file_path}\n')

                f.write('Validation indices and paths:\n')
                for idx in self.val_indices[i]:
                    class_name, recording_name = self.all_recording_names[idx]
                    file_paths = [file_data['file_path'] for file_data in self.organized_data[class_name][recording_name]]
                    for file_path in file_paths:
                        f.write(f'{idx}: {file_path}\n')

                f.write('\n')

    def train_dataloader(self):
        train_dataset = AudioDataset(self.train_data, self.class_to_idx)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        val_dataset = AudioDataset(self.val_data, self.class_to_idx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)