
from __future__ import print_function
from __future__ import division
import numpy as np
import argparse

# PyTorch dependencies
import torch

# Local external libraries
from Demo_Parameters import Parameters
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from Datasets.Get_preprocessed_data import process_data

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from Datasets.SSDataModule import SSAudioDataModule
from Utils.LitModel import LitModel
    
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

    new_dir = Params["new_dir"]  
    
    process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    print("\nModel name: ", model_name, "\n")
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
    data_module.prepare_data()
    
    s_rate=Params['sample_rate']

    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    all_test_accs = []
    
    for run_number in range(numRuns):
        if run_number != 0:
            seed_everything(run_number + 1, workers=True)
    
        print(f'\nStarting Run {run_number}')
    
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
            f"tb_logs/{model_name}_b{batch_size}_{s_rate}/Run_{run_number}",
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
    
        best_test_acc = test_results[0]['test_acc']
        all_test_accs.append(best_test_acc)
    
        results_filename = f"tb_logs/{model_name}_b{batch_size}_{s_rate}/Run_{run_number}/metrics.txt"
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}:\n\n")
            file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            file.write(f"Best Test Accuracy: {best_test_acc:.4f}\n")
    
    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
    
    overall_avg_test_acc = np.mean(all_test_accs)
    overall_std_test_acc = np.std(all_test_accs)
    
    summary_filename = f"tb_logs/{model_name}_b{batch_size}_{s_rate}/summary_metrics.txt"
    with open(summary_filename, "w") as file:
        file.write("Overall Results Across All Runs\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n\n")
    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run experiments')
    parser.add_argument('--model', type=str, default='CNN_14_32k', #CNN_14_16k #convnextv2_tiny.fcmae 
                        help='Select baseline model architecture')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Select optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
