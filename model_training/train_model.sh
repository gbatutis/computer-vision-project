#!/bin/bash

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_resnet --batch_size 64 --lr 0.01 --momentum 0.9 --step_size 7 --gamma 0.1 --epochs 2
