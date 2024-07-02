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
import pdb

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


from SSDataModule import SSAudioDataModule

from Utils.Network_functions import CustomPANN, initialize_model, download_weights, set_parameter_requires_grad
from torchmetrics.classification import F1Score

from LitModel import LitModel

def main(Params):

    # Name of dataset
    Dataset = Params['Dataset']

    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]

    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\nStarting Experiments...')
    
    numRuns = 3
    run_number = 0
    seed_everything(run_number+1, workers=True)

    data_dir = Params["data_dir"]  
    new_dir = Params["new_dir"]  
    
    process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    print("\nDataset sample rate: ", Params['sample_rate'])
    print("\nModel name: ", model_name, "\n")
    
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
    data_module.prepare_data()


    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    all_test_accs = []
    all_test_f1s = []
    
    for run_number in range(numRuns):
        if run_number != 0:
            seed_everything(run_number + 1, workers=True)
    
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
            pretrained_loaded=False,
            run_number=run_number
        )
    
        logger = TensorBoardLogger(
            f"tb_logs/{model_name}_b{batch_size}_SS/Run_{run_number}",
            name=f"{model_name}"
        )
    
        trainer = Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=logger
        )
    
        trainer.fit(model=model_AST, datamodule=data_module)
    
        best_val_acc = checkpoint_callback.best_model_score.item()
        all_val_accs.append(best_val_acc)
    
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
            Dataset=Dataset,
            pretrained_loaded=True,
            run_number=run_number
        )
    
        test_results = trainer.test(model=best_model, datamodule=data_module)
        best_test_f1 = max(result['test_f1'] for result in test_results)
        all_test_f1s.append(best_test_f1)
    
        best_test_acc = max(result['test_acc'] for result in test_results)
        all_test_accs.append(best_test_acc)
    
        results_filename = f"tb_logs/{model_name}_b{batch_size}_SS/Run_{run_number}/metrics.txt"
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}:\n\n")
            file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            file.write(f"Best Test Accuracy: {best_test_acc:.4f}\n")
            file.write(f"Best Test F1 Score: {best_test_f1:.4f}\n\n")
    
    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
    
    overall_avg_test_acc = np.mean(all_test_accs)
    overall_std_test_acc = np.std(all_test_accs)
    
    overall_avg_test_f1 = np.mean(all_test_f1s)
    overall_std_test_f1 = np.std(all_test_f1s)
    
    summary_filename = f"tb_logs/{model_name}_b{batch_size}_SS/summary_metrics.txt"
    with open(summary_filename, "w") as file:
        file.write("Overall Results Across All Runs\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test F1 Scores: {overall_avg_test_f1:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test F1 Scores: {overall_std_test_f1:.4f}\n\n")
    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='efficientnet_b3', #CNN_14_16k #CNN_14_16k #ViT-B/16
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
    parser.add_argument('--train_batch_size', type=int, default=128,
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
    parser.add_argument('--patience', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
