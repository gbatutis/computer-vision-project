#!/bin/bash

python train_model_arch_adj.py --train_images 'train_data_2sats.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2sats.npz' --val_labels 'val_labels2.npz' --train_senet_newarch --batch_size 64 --lr 0.03 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20
