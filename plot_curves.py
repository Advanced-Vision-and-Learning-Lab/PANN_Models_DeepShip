
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from Utils.RBFHistogramPooling import HistogramLayer
import torch

from Prepare_Data import Prepare_DataLoaders  
import numpy as np

import argparse
from Demo_Parameters import Parameters

import lightning as L
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Utils.Network_functions import initialize_model

from demo_light import LitModel
from torch.cuda.amp import autocast


def extract_metric(logdir, tag):
    event_file = [os.path.join(logdir, f) for f in os.listdir(logdir) if 'tfevents' in f]
    if not event_file:
        print(f"No event files found in {logdir}")
        return []
    event_file = event_file[0]

    ea = event_accumulator.EventAccumulator(event_file,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' not found in {logdir}")
        return []

    events = ea.Scalars(tag)
    return [event.value for event in events]

def plot_metrics(metrics, fold_labels, title, filename):
    plt.figure(figsize=(10, 8), dpi=300)
    for i, metric in enumerate(metrics):
        if metric:
            plt.plot(metric, label=f'Fold {fold_labels[i]}')
        else:
            print(f"No data for Fold {fold_labels[i]}")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title.split(' ')[0])  
    plt.legend()
    plt.grid(True)
    filename = 'tb_logs/Figures/' + filename 
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()


def compute_confusion_matrices(predictions, labels, num_classes):
    matrices = []
    for preds, labs in zip(predictions, labels):
        cm = confusion_matrix(labs, preds, labels=range(num_classes))
        matrices.append(cm)
    return matrices


def get_predictions_labels(base_path, num_runs, num_folds, Params):
    all_predictions = []
    all_labels = []

    model_name=params['Model_name']
    Dataset = params['Dataset']
    num_classes = params['num_classes'][Dataset]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    for run in range(num_runs):
        run_predictions = []
        run_labels = []
        for fold_index in range(num_folds):
            checkpoint_folder = f"{base_path}/Run_{run}/AST_fold_{fold_index}_48x80/version_0/checkpoints"
            checkpoint_file = next(file for file in os.listdir(checkpoint_folder) if file.endswith('.ckpt'))
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
            
            model = LitModel.load_from_checkpoint(checkpoint_path,HistogramLayer=HistogramLayer, Params=params, model_name=params['Model_name'],
                             num_classes=num_classes, num_feature_maps=params['out_channels'][model_name],
                             feat_map_size=params['feat_map_size'], numBins=params['numBins'], Dataset=params['Dataset'])
            
            model = model.to(device)
            model.eval()

            dataloaders_dict = Prepare_DataLoaders(Params, fold_index, num_folds=num_folds)
            val_loader = dataloaders_dict['val']

            fold_predictions, fold_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)  
                    y = y.to(device)
                    y_hat = model(x)
                    preds = torch.argmax(y_hat, dim=1)
                    
                    fold_predictions.extend(preds.cpu().numpy())
                    fold_labels.extend(y.cpu().numpy())

            run_predictions.append(fold_predictions)
            run_labels.append(fold_labels)
        
        all_predictions.append(run_predictions)
        all_labels.append(run_labels)
    
    return all_predictions, all_labels

def save_confusion_matrices(cm_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cm_list, f)
            
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')
 
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


    base_dir = 'tb_logs/STFT_b256_AST_shFalse'
    
    fold_labels = range(3)
    tags = {
        'Training Loss': 'loss_epoch',
        'Training Accuracy': 'train_acc_epoch',
        'Validation Loss': 'val_loss',
        'Validation Accuracy': 'val_acc'
    }

    for tag_name, tag in tags.items():
        metrics = []
        for i in fold_labels:
            log_dir = os.path.join(base_dir, f'Run_0/AST_fold_{i}_48x80', 'version_0')
            metric = extract_metric(log_dir, tag)
            metrics.append(metric)
        plot_metrics(metrics, fold_labels, f'{tag_name} Across Folds', f'{tag.lower().replace(" ", "_")}.png')



    base_path = 'tb_logs/STFT_b256_AST_shFalse'

    num_runs = 3
    num_folds = 3
    
    all_predictions, all_labels = get_predictions_labels(base_path, num_runs, num_folds, Params)
    
    flat_predictions = [item for sublist in all_predictions for item in sublist]
    flat_labels = [item for sublist in all_labels for item in sublist]
    
    cm_list = compute_confusion_matrices(flat_predictions, flat_labels, num_classes=num_classes)
    
    save_confusion_matrices(cm_list, 'tb_logs/CM_Matrices/confusion_matrices_STFT_b64.pkl')


    

        
def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/lightning/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='AST',
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
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                       help = 'Select optimizer')
    parser.add_argument('--patience', type=int, default=8,
                        help='Number of epochs to train each model for (default: 50)')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # use_cuda = args.use_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)



