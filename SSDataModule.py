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
from sklearn.model_selection import StratifiedShuffleSplit


class SSAudioDataset(Dataset):
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


class SSAudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=256, test_size=0.2, val_size=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.class_to_idx = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self.prepared = False

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

    def create_splits(self, organized_data):
        all_recording_names = []
        class_labels = []

        for class_name, recordings in organized_data.items():
            for recording_name in recordings.keys():
                all_recording_names.append((class_name, recording_name))
                class_labels.append(class_name)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, random_state=0)
        train_index, temp_index = next(sss.split(all_recording_names, class_labels))

        val_test_size = self.val_size / (self.test_size + self.val_size)
        sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=val_test_size, random_state=0)
        val_index, test_index = next(sss_temp.split(np.array(all_recording_names)[temp_index], np.array(class_labels)[temp_index]))

        train_data, val_data, test_data = [], [], []

        for idx in train_index:
            class_name, recording_name = all_recording_names[idx]
            train_data.extend(organized_data[class_name][recording_name])

        for idx in val_index:
            class_name, recording_name = all_recording_names[temp_index[idx]]
            val_data.extend(organized_data[class_name][recording_name])

        for idx in test_index:
            class_name, recording_name = all_recording_names[temp_index[idx]]
            test_data.extend(organized_data[class_name][recording_name])

        print(f'Created train, validation, and test splits')
        return train_data, val_data, test_data

    def check_data_leakage(self):
        print("\nChecking data leakage")

        all_data = self.train_data + self.val_data + self.test_data
        flattened_data = [item for sublist in all_data for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Debugging: print the structure of flattened_data
        #print("Sample of flattened_data structure:", flattened_data[0] if flattened_data else "No data")

        # Ensure flattened_data is a list of dictionaries with 'file_path' key
        if not isinstance(flattened_data, list):
            raise ValueError("flattened_data should be a list")
        if not all(isinstance(file_data, dict) for file_data in flattened_data):
            raise ValueError("Each element in flattened_data should be a dictionary")
        if not all('file_path' in file_data for file_data in flattened_data):
            raise ValueError("Each dictionary in flattened_data should contain the 'file_path' key")

        file_paths = [file_data['file_path'] for file_data in flattened_data]
        unique_file_paths = set(file_paths)

        if len(file_paths) != len(unique_file_paths):
            print("\nData leakage detected: Some samples are present in more than one split!\n")

            # Identify and print the duplicated file paths
            from collections import Counter
            file_path_counts = Counter(file_paths)
            duplicated_paths = [file_path for file_path, count in file_path_counts.items() if count > 1]

            print("\nDuplicated file paths:")
            for path in duplicated_paths:
                print(path)
        else:
            print("\nNo data leakage detected.\n")

    def count_samples_per_class(self, data_list):
        class_counts = defaultdict(int)
        for file_data in data_list:
            class_name = file_data['file_path'].split(os.sep)[-3]
            class_counts[class_name] += 1
        return class_counts

    def print_class_distribution(self):
        print('Train set class distribution:')
        train_class_counts = self.count_samples_per_class(self.train_data)
        for class_name, count in train_class_counts.items():
            print(f'  {class_name}: {count}')

        print('Validation set class distribution:')
        val_class_counts = self.count_samples_per_class(self.val_data)
        for class_name, count in val_class_counts.items():
            print(f'  {class_name}: {count}')

        print('Test set class distribution:')
        test_class_counts = self.count_samples_per_class(self.test_data)
        for class_name, count in test_class_counts.items():
            print(f'  {class_name}: {count}')

    def get_min_max_train(self):
        global_min = float('inf')
        global_max = float('-inf')
        for file_data in self.train_data:
            data = file_data['data'].astype(np.float32)
            file_min = np.min(data)
            file_max = np.max(data)
            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max
        return global_min, global_max

    def normalize_data(self, data_list, global_min, global_max):
        print("Normalizing train/val/test")
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

    def save_split_indices(self, filepath):
        print("\nSaving split indices")
        with open(filepath, 'w') as f:
            f.write('Train indices and paths:\n')
            for idx, file_data in enumerate(self.train_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')
    
            f.write('\nValidation indices and paths:\n')
            for idx, file_data in enumerate(self.val_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')
    
            f.write('\nTest indices and paths:\n')
            for idx, file_data in enumerate(self.test_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')


    def prepare_data(self):
        if not self.prepared:
            self.wav_files = self.list_wav_files()
            self.data_list = self.read_wav_files(self.wav_files)
            self.organized_data = self.organize_data(self.data_list)
            self.train_data, self.val_data, self.test_data = self.create_splits(self.organized_data)
            self.check_data_leakage()
            self.print_class_distribution()
            
            # Get global min and max values from training data
            self.global_min, self.global_max = self.get_min_max_train()
            
            # Normalize train, validation, and test data using the global min and max
            self.train_data = self.normalize_data(self.train_data, self.global_min, self.global_max)
            self.val_data = self.normalize_data(self.val_data, self.global_min, self.global_max)
            self.test_data = self.normalize_data(self.test_data, self.global_min, self.global_max)
            
            self.prepared = True
    
    def setup(self, stage=None):
        pass


    def train_dataloader(self):
        train_dataset = SSAudioDataset(self.train_data, self.class_to_idx)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        val_dataset = SSAudioDataset(self.val_data, self.class_to_idx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        test_dataset = SSAudioDataset(self.test_data, self.class_to_idx)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
