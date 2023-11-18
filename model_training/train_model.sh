#!/bin/bash

python train_model.py --train_images 'val_data.npz' --train_labels 'val_labels.npz' --val_images 'val_data.npz' --val_labels 'val_labels.npz' --train_resnet --batch_size 64 --lr 0.01 --momentum 0.9 --step_size 7 --gamma 0.1 --epochs 2
