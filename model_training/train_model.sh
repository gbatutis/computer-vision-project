#!/bin/bash

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_resnet --batch_size 16 --lr 0.01 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20; 

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_resnet --batch_size 32 --lr 0.01 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20; 

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_resnet --batch_size 96 --lr 0.01 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_senet --batch_size 16 --lr 0.03 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20;

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_senet --batch_size 32 --lr 0.03 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20;

python train_model.py --train_images 'train_data_final.npz' --train_labels 'train_labels.npz' --val_images 'val_data_2.npz' --val_labels 'val_labels2.npz' --train_senet --batch_size 96 --lr 0.03 --momentum 0.9 --step_size 15 --gamma 0.3 --epochs 20
