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
from Utils.Network_functions import initialize_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Utils.Get_Optimizer import get_optimizer
from Demo_Parameters import Parameters

# from Prepare_Data import Prepare_DataLoaders
# from Prepare_Data import read_wav_files, organize_data, create_stratified_k_folds,get_min_max_train
# from Prepare_Data import list_wav_files, count_samples_per_class, check_data_leakage
#from Datasets.DeepShipSegments import DeepShipSegments

from Utils.TDNN import TDNN
import torch.nn.functional as F
import os
from Utils.Save_Results import get_file_location

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import tqdm

import pdb

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import torchmetrics
from lightning.pytorch.callbacks import TQDMProgressBar

from torchmetrics import Accuracy

from Datasets.Get_preprocessed_data import process_data

from tqdm.auto import tqdm
from lightning.pytorch.callbacks import Callback
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from KFoldDataModule import AudioDataModule

class LitModel(L.LightningModule):

    def __init__(self, HistogramLayer, Params, model_name, num_classes, num_feature_maps, feat_map_size, numBins, Dataset):
        super().__init__()

        self.learning_rate = Params['lr']

        # histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
        #                                  Params['kernel_size'][model_name], dim=1,
        #                                  num_bins=numBins, stride=Params['stride'],
        #                                  normalize_count=Params['normalize_count'],
        #                                  normalize_bins=Params['normalize_bins'])


        histogram_layer = HistogramLayer(768,
                                          Params['kernel_size'][model_name], dim=1,
                                          num_bins=numBins, stride=Params['stride'],
                                          normalize_count=Params['normalize_count'],
                                          normalize_bins=Params['normalize_bins'])

        self.model_ft, input_size, self.feature_extraction_layer, self.ft_dims = initialize_model(model_name, num_classes,
                                                                                                  Params['in_channels'][model_name],
                                                                                                  # len(Params['feature']),
                                                                                                  num_feature_maps,
                                                                                                  feature_extract=Params[
                                                                                                      'feature_extraction'],
                                                                                                  histogram=Params['histogram'],
                                                                                                  histogram_layer=histogram_layer,
                                                                                                  parallel=Params['parallel'],
                                                                                                  use_pretrained=Params[
                                                                                                      'use_pretrained'],
                                                                                                  add_bn=Params['add_bn'],
                                                                                                  scale=Params['scale'],
                                                                                                  feat_map_size=feat_map_size,
                                                                                                  TDNN_feats=(
                                                                                                      Params['TDNN_feats'][Dataset]),
                                                                                                  input_feature=Params['feature'])

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()
        self.first_epoch_time_start = None

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        return y_pred

    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = train_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)

        self.train_acc(y_pred, y)

        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.first_epoch_time_start = time.time()

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        self.log('train_acc', train_acc)
        print(f'Training Accuracy: {train_acc:.4f}')
        self.train_acc.reset()

        if self.current_epoch == 0 and self.first_epoch_time_start is not None:
            epoch_duration = time.time() - self.first_epoch_time_start
            print(f"Duration of the first epoch: {epoch_duration:.2f} seconds")


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        val_loss = F.cross_entropy(y_pred, y)

        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        return val_loss

    def on_validation_epoch_end(self):
        # Compute and log the validation accuracy only once at the end of the epoch
        val_acc = self.val_acc.compute()
        self.log('val_acc', val_acc)
        print(f'Validation Accuracy: {val_acc:.4f}')
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
# Custom callback to measure and print the training time for one epoch
class TimeEpochCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"Epoch {trainer.current_epoch} duration: {duration:.2f} seconds")




def main(Params):

    # Name of dataset
    Dataset = Params['Dataset']

    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]

    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]

    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']
    kernel_size = Params['kernel_size'][model_name]
    in_channels = Params['in_channels'][model_name]

    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\nStarting Experiments...')
    
    run_number = 0
    seed_everything(run_number, workers=True)
    all_runs_accs = []
    num_folds = 3
    
    data_dir = Params["data_dir"]  
    new_dir = Params["new_dir"]  
    
    process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    
    data_module = AudioDataModule(new_dir, batch_size=batch_size, num_folds=num_folds)
    data_module.prepare_data()
    data_module.save_fold_indices('kfold_data_split.txt')  

    torch.set_float32_matmul_precision('medium')
    for run_number in range(0, numRuns-2):
        best_val_accs = []
        
        if run_number != 0:
            seed_everything(run_number, workers=True)
                 
        for fold_index in range(num_folds):            
            print(f'\nStarting Run {run_number}, Fold {fold_index}\n')

            data_module.set_fold_index(fold_index)
                        
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

            model_AST = LitModel(HistogramLayer, Params, model_name, num_classes, num_feature_maps,
                                 feat_map_size, numBins, Dataset)

            if model_AST.ft_dims is not None and len(model_AST.ft_dims) > 1:
                dim_str = 'x'.join(map(str, model_AST.ft_dims[1:]))
            else:
                dim_str = 'unknown_dims'

            logger = TensorBoardLogger(
                f"tb_logs/{Params['feature']}_b{batch_size}_{Params['Model_name']}/Run_{run_number}",
                name=f"{Params['Model_name']}_fold_{fold_index}_{dim_str}"
            )

            trainer = L.Trainer(
                max_epochs=Params['num_epochs'],
                callbacks=[early_stopping_callback, checkpoint_callback],
                deterministic=False,
                logger=logger
            )

            trainer.fit(model=model_AST, datamodule=data_module)

            best_val_accs.append(checkpoint_callback.best_model_score.item())

        average_val_acc = np.mean(best_val_accs)
        std_val_acc = np.std(best_val_accs)
        all_runs_accs.append(best_val_accs)

        results_filename = f"tb_logs/{Params['feature']}_b{batch_size}_{Params['Model_name']}/Run_{run_number}/{Params['feature']}_{dim_str}.txt"
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}_{Params['feature']}_{dim_str}\n")
            file.write(
                f"Average of Best Validation Accuracy: {average_val_acc:.4f}\n")
            file.write(
                f"Standard Deviation of Best Validation Accuracies: {std_val_acc:.4f}\n\n")

    # Flatten the list of lists and compute overall statistics
    flat_list = [acc for sublist in all_runs_accs for acc in sublist]
    overall_avg_acc = np.mean(flat_list)
    overall_std_acc = np.std(flat_list)

    summary_filename = f"tb_logs/{Params['feature']}_b{batch_size}_{Params['Model_name']}/summary_results.txt"
    with open(summary_filename, "w") as file:
        file.write(
            f"Overall Results Across All Runs for {Params['feature']}\n")
        file.write(
            f"Overall Average of Best Validation Accuracies: {overall_avg_acc:.4f}\n")
        file.write(
            f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_acc:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='CNN_14',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=False, action=argparse.BooleanOptionalAction,
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
