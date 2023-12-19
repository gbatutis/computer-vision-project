import torch
import numpy as np

loaded_data = np.load("/scratch/gsb8013/cv-final-project/model_training/val_data_2sats.npz")
input_tensor = loaded_data["arr_0"]

# Create the first tensor of size (1000, 9, 50, 50)
tensor_part1 = input_tensor[:, :9, :, :]

print("tensor part 1 size: ", tensor_part1.shape)

# Create the second tensor of size (1000, 48, 50, 50)
tensor_part2 = input_tensor[:, 9:, :, :]

# Print the sizes of the resulting tensors
#print("Tensor Part 1 Size:", tensor_part1.shape)
print("Tensor Part 2 Size:", tensor_part2.shape)

# Convert augmented_data back to numpy array
arr_0 = tensor_part1

# Save the augmented data to an npz file
np.savez("val_electricity.npz", arr_0)

# Convert augmented_data back to numpy array
arr_0 = tensor_part2

# Save the augmented data to an npz file
np.savez("val_humans.npz", arr_0)
