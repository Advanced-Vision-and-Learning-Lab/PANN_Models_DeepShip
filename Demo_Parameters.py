# -*- coding: utf-8 -*-
"""
Parameters for experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import os
import sys

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ########

    #optimizer selection
    optimizer = args.optimizer

    #Select dataset. Set to number of desired dataset
    data_selection = args.data_selection
    Dataset_names = {0: 'DeepShip'}
    
    #Flag for feature extraction. False, train whole model. True, only update
    #fully connected and histogram layers parameters (default: False)
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    #Flag to add BN to convolutional features (default:True)
    #Location/Scale at which to apply histogram layer (default: 5 (at the end))
    feature_extraction = args.feature_extraction
    use_pretrained = args.use_pretrained
    add_bn = True
    
    #Set learning rate for new layers
    #Recommended values are .01 (used in paper) or .001
    lr = args.lr
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size} 
    num_epochs = args.num_epochs
    
    #Patience is the number of epochs to observe if a metric (loss or accuarcy)
    #is minimized or maximized
    patience = args.patience
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = False
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 16
    
    #Select audio feature for DeepShip 
    feature = args.audio_feature
    
    #Set to True if more than one GPU was used
    Parallelize_model = True
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    #Location of texture datasets
    Data_dirs = {'DeepShip': './Datasets/DeepShip/Segments/'}
    
    segment_length = {'DeepShip': 5}
    #sample_rate ={'DeepShip': 32000}
    sample_rate = args.sample_rate
    
    #ResNet models to use for each dataset
    Model_name = args.model
    
    #Number of classes in each dataset
    num_classes = {'DeepShip': 4}
    
    #Number of runs and/or splits for each dataset
    Splits = {'DeepShip': 3}   

    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    segment_length = segment_length[Dataset]

    new_dir_p = './Datasets/DeepShip/'
    new_dir = '{}Segments_{}s_{}hz/'.format(new_dir_p,segment_length,sample_rate)
    
    #Save results based on features (can adapt for additional audio datasets or computer vision datasets)
    if (Dataset=='DeepShip'):
        audio_features = True
    else:
        audio_features = False
    
    #Return dictionary of parameters
    Params = {'Dataset': Dataset, 'data_dir': data_dir,
                          'segment_length':segment_length,'sample_rate':sample_rate,
                          'optimizer': optimizer,'new_dir': new_dir,
                          'num_workers': num_workers, 'mode': mode,'lr': lr,
                          'step_size': step_size, 'gamma': gamma, 'batch_size' : batch_size, 
                          'num_epochs': num_epochs,             
                          'Model_name': Model_name, 'num_classes': num_classes, 
                          'Splits': Splits, 'feature_extraction': feature_extraction, 
                          'use_pretrained': use_pretrained,
                          'add_bn': add_bn, 'pin_memory': pin_memory, 
                          'feature': feature, 'audio_features': audio_features,
                          'patience': patience}
    return Params

