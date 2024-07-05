#!/bin/bash

python demo_light.py --model CNN_14_32k --train_batch_size 64 --num_epochs 100 --sample_rate 32000
python demo_light.py --model CNN_14_32k --train_batch_size 64 --num_epochs 100 --sample_rate 8000


