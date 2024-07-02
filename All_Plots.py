#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:38:19 2024

@author: amir.m
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import os


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

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.manifold import TSNE

def save_tsne_plot(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            features, _ = model(x)
            all_features.append(features.cpu())
            all_labels.append(y.cpu())

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        indices = all_labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=class_name, alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Plot')
    plt.legend()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def save_confusion_matrix(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, y_pred = model(x)
            all_preds.append(y_pred.cpu().argmax(dim=1))
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_multiclass_roc(model, test_loader, class_names, device, output_path):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    num_classes = len(class_names)

    # Binarize the output
    all_labels = np.eye(num_classes)[all_labels]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-class')
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()



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

    data_dir = Params["data_dir"]  
    new_dir = Params["new_dir"]  
    
    print("\nDataset sample rate: ", Params['sample_rate'])
    print("\nModel name: ", model_name, "\n")
    
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
    data_module.prepare_data()


    torch.set_float32_matmul_precision('medium')

    
    best_model_path = f"tb_logs/{model_name}_b{batch_size}_SS/Run_{run_number}/{model_name}/version_0/checkpoints/best-epoch=00-val_acc=0.64.ckpt"


    best_model = LitModel.load_from_checkpoint(
        checkpoint_path=best_model_path,
        Params=Params,
        model_name=model_name,
        num_classes=num_classes,
        Dataset=Dataset,
        pretrained_loaded=True,
        run_number=run_number
    )
        

    # Get the test dataloader from the data module
    test_loader = data_module.test_dataloader()
    
    # Define class names
    class_names = ["Cargo", "Passengership", "Tanker", "Tug"]
    
    # Define output path for the ROC plot
    output_path = "features/roc_curve.png"
    
    # Plot ROC curves and save the figure
    plot_multiclass_roc(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=output_path)
        
    # Define output path for the confusion matrix plot
    cm_output_path = "features/confusion_matrix.png"
    
    # Save confusion matrix
    save_confusion_matrix(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=cm_output_path)
      
    # Define output path for the t-SNE plot
    tsne_output_path = "features/tsne_plot.png"
    
    # Save t-SNE plot
    save_tsne_plot(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=tsne_output_path)
        

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='CNN_14_32k', #CNN_14_16k #CNN_14_16k #ViT-B/16
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
