#!/bin/bash

python demo_light.py --model CNN_14_8k --train_batch_size 32 --lr 5e-5 --num_epochs 100 --patience 50 --num_epochs 1 --sample_rate 4000
#python demo_light.py --model CNN_14_32k --train_batch_size 32 --lr 5e-5 --num_epochs 100 --patience 50 --num_epochs 150 --sample_rate 16000
#python demo_light.py --model resnet50 --train_batch_size 64 --lr 5e-5 --num_epochs 100 --patience 50 --num_epochs 150 --sample_rate 32000

