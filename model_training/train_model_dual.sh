#!/bin/bash

python train_model.py --train_images 'train_data_2sats.npz' --train_labels 'train_labels_humans.npz' --val_images 'val_data_2sats.npz' --val_labels 'val_labels_humans.npz' --train_senet --humans --batch_size 64 --lr 0.01 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20;

python train_model.py --train_images 'train_data_2sats.npz' --train_labels 'train_labels_humans.npz' --val_images 'val_data_2sats.npz' --val_labels 'val_labels_humans.npz' --train_senet --humans --batch_size 64 --lr 0.005 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20;
