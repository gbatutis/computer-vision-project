#!/bin/bash

python train_model.py --train_images 'train_data_2sats.npz' --train_labels 'train_labels_electricity.npz' --val_images 'val_data_2sats.npz' --val_labels 'val_labels_electricity.npz' --train_senet --elec --batch_size 32 --lr 0.03 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20;
