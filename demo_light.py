#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:01:06 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import random

# PyTorch dependencies
import torch
import torch.nn as nn

# Local external libraries
from Demo_Parameters import Parameters


import torch.nn.functional as F
import os

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import torchmetrics

from Datasets.Get_preprocessed_data import process_data

from tqdm.auto import tqdm

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from KFoldDataModule import AudioDataModule
from SSDataModule import SSAudioDataModule

from Utils.Network_functions import CustomPANN, initialize_model, download_weights, set_parameter_requires_grad
from torchmetrics.classification import F1Score


class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, Dataset, pretrained_loaded):
        super().__init__()

        self.learning_rate = Params['lr']

        self.model_ft, input_size, self.mel_extractor = initialize_model(
            model_name, 
            use_pretrained=Params['use_pretrained'], 
            feature_extract=Params['feature_extraction'], 
            num_classes=num_classes,
            pretrained_loaded=pretrained_loaded 
        )

        self.save_hyperparameters()
        
        self.test_preds = []
        self.test_labels = []
        self.test_features = []

        # Initialize accuracy metrics
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        # Extract mel spectrogram if not PANN model
        x = self.mel_extractor(x)
        # features are from the feature layer (backbone)
        features, y_pred = self.model_ft(x)
        return features, y_pred


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        features, y_pred = self(x) 
        loss = F.cross_entropy(y_pred, y)
        
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)
        
        return loss


    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        self.log('train_acc_epoch', train_acc)
        print(f'Training Accuracy: {train_acc:.4f}')
        self.train_acc.reset()


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        features, y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
    
        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
    
        return val_loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.log('val_acc_epoch', val_acc)
        print(f'Validation Accuracy: {val_acc:.4f}')
        self.val_acc.reset()
 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        features, y_pred = self(x)
        test_loss = F.cross_entropy(y_pred, y)
    
        self.test_acc(y_pred, y)
        self.test_f1(y_pred, y)  # Update F1 score
    
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)  # Log F1 score
    
        # Store predictions, true labels, and features
        self.test_preds.append(y_pred.cpu())
        self.test_labels.append(y.cpu())
        self.test_features.append(features.cpu())
    
        return test_loss
    
    def on_test_epoch_end(self):
        # Compute and log the test accuracy only once at the end of the epoch
        test_acc = self.test_acc.compute()
        self.log('test_acc_epoch', test_acc)
        print(f'Test Accuracy: {test_acc:.4f}')
        self.test_acc.reset()
    
        # Compute and log the F1 score for the test set
        test_f1 = self.test_f1.compute()
        self.log('test_f1_epoch', test_f1)
        print(f'Test F1 Score: {test_f1:.4f}')
        self.test_f1.reset()    
    
        # Save features and labels with model name
        self.save_features("test", self.hparams.model_name)
    
    
    def save_features(self, phase, model_name):
        # Define file paths to save features, predictions, and labels
        os.makedirs("features", exist_ok=True)
        features_file_path = f"features/{model_name}_{phase}_features.pth"
        labels_file_path = f"features/{model_name}_{phase}_labels.pth"
        preds_file_path = f"features/{model_name}_{phase}_preds.pth"
    
        # Convert lists to tensors
        features_tensor = torch.cat(self.test_features)
        labels_tensor = torch.cat(self.test_labels)
        preds_tensor = torch.cat(self.test_preds).argmax(dim=1)  # Convert logits to class labels
    
        # Save tensors
        torch.save({'features': features_tensor, 'labels': labels_tensor, 'preds': preds_tensor}, features_file_path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main(Params):

    # Name of dataset
    Dataset = Params['Dataset']

    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]

    # Local area of feature map after histogram layer
    #feat_map_size = Params['feat_map_size']
    #kernel_size = Params['kernel_size'][model_name]
    #in_channels = Params['in_channels'][model_name]

    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\nStarting Experiments...')
    
    
    numRuns = 1
    run_number = 0
    seed_everything(run_number, workers=True)

    data_dir = Params["data_dir"]  
    new_dir = Params["new_dir"]  
    
    process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    print("\nDataset sample rate: ", Params['sample_rate'])
    print("\nModel name: ", model_name, "\n")
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size)
    data_module.prepare_data()
    data_module.save_split_indices('split_indices.txt')
    
    torch.set_float32_matmul_precision('medium')
    all_runs_val_accs = []
    all_runs_test_accs = []
    all_runs_test_f1s = []  # Initialize list to store F1 scores
    for run_number in range(0, numRuns):
        pretrained_loaded = False
        best_val_accs = []
        best_test_accs = []
        best_test_f1s = []  # Initialize list for best F1 scores
    
        if run_number != 0:
            seed_everything(run_number, workers=True)
    
        print(f'\nStarting Run {run_number}\n')
    
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max',
            verbose=True,
            save_weights_only=True
        )
    
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=Params['patience'],
            verbose=True,
            mode='min'
        )
    

        model_AST = LitModel(
        Params=Params, 
        model_name=model_name, 
        num_classes=num_classes, 
        Dataset=Dataset, 
        pretrained_loaded=False  
        )
    
    
        logger = TensorBoardLogger(
            f"tb_logs/{Params['Model_name']}_b{batch_size}_SS/Run_{run_number}",
            name=f"{Params['Model_name']}"
        )
    
        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=logger
        )
    
        trainer.fit(model=model_AST, datamodule=data_module)
        
        best_val_accs.append(checkpoint_callback.best_model_score.item())
    
        # Load best model checkpoint for testing
        best_model_path = checkpoint_callback.best_model_path
        #print(best_model_path)
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path, 
            Params=Params, 
            model_name=model_name, 
            num_classes=num_classes, 
            Dataset=Dataset,
            pretrained_loaded=True  
        )
        
        # Test the best model
        test_results = trainer.test(model=best_model, datamodule=data_module)

        best_test_f1 = max(result['test_f1'] for result in test_results)
        best_test_f1s.append(best_test_f1)
        
        best_test_acc = max(result['test_acc'] for result in test_results)
        best_test_accs.append(best_test_acc)
    
        average_val_acc = np.mean(best_val_accs)
        std_val_acc = np.std(best_val_accs)
        all_runs_val_accs.append(best_val_accs)
    
        average_test_acc = np.mean(best_test_accs)
        std_test_acc = np.std(best_test_accs)
        all_runs_test_accs.append(best_test_accs)
    
        average_test_f1 = np.mean(best_test_f1s)
        std_test_f1 = np.std(best_test_f1s)
        all_runs_test_f1s.append(best_test_f1s)   
            

        results_filename = f"tb_logs/{Params['Model_name']}_b{batch_size}_SS/Run_{run_number}/{Params['Model_name']}.txt"
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}\n")
            file.write(f"Average of Best Validation Accuracy: {average_val_acc:.4f}\n")
            file.write(f"Standard Deviation of Best Validation Accuracies: {std_val_acc:.4f}\n\n")
            file.write(f"Average of Best Test Accuracy: {average_test_acc:.4f}\n")
            file.write(f"Standard Deviation of Best Test Accuracies: {std_test_acc:.4f}\n\n")
            file.write(f"Average of Best Test F1 Score: {average_test_f1:.4f}\n")
            file.write(f"Standard Deviation of Best Test F1 Scores: {std_test_f1:.4f}\n\n")
    
    # Flatten the list of lists and compute overall statistics
    flat_val_list = [acc for sublist in all_runs_val_accs for acc in sublist]
    overall_avg_val_acc = np.mean(flat_val_list)
    overall_std_val_acc = np.std(flat_val_list)
    
    flat_test_list = [acc for sublist in all_runs_test_accs for acc in sublist]
    overall_avg_test_acc = np.mean(flat_test_list)
    overall_std_test_acc = np.std(flat_test_list)

    flat_test_f1_list = [f1 for sublist in all_runs_test_f1s for f1 in sublist]
    overall_avg_test_f1 = np.mean(flat_test_f1_list)
    overall_std_test_f1 = np.std(flat_test_f1_list)
    
    summary_filename = f"tb_logs/{Params['Model_name']}_b{batch_size}_SS/summary_results.txt"
    with open(summary_filename, "w") as file:
        file.write("Overall Results Across All Runs\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n")
        file.write(f"Overall Average of Best Test F1 Scores: {overall_avg_test_f1:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test F1 Scores: {overall_std_test_f1:.4f}\n")



def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='CNN_14_16k',
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
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to train each model for (default: 50)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # use_cuda = args.use_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)
