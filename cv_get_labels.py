import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from tqdm import tqdm

all_labels = []

# Process 60 tiles
for tile_number in tqdm(range(1, 61)):
    # Construct the file path for each tile
    tile_path = f"/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/dfc2021_dse_val_reference/{tile_number}.tif"

    # Read the tile image
    label_image = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)

    # Flatten the array and adjust values
    flattened_array = label_image.flatten()
    labels = np.array(flattened_array) - 1  # Convert from 1-4 to 0-3

    # Append the flattened array to the list
    all_labels.append(labels)

# Concatenate all the flattened arrays into one long array
final_labels = np.concatenate(all_labels)
print("final labels size",final_labels.size)
print(final_labels)

#FIND CLASS IMBALANCE!!
# Get unique values and their counts
# unique_values, counts = np.unique(final_labels, return_counts=True)

# # Print class imbalance
# for value, count in zip(unique_values, counts):
#     print(f"Class {value}: {count} samples")

#save the data
np.savez_compressed("/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/val_labels2.npz",labels=final_labels) #stacked_tensor=stacked_tensor, 
print("complete")