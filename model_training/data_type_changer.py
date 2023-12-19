import numpy as np

# Load the .npz file
file_path = 'train_data_2sats.npz'
archive = np.load(file_path)

# Access the array you want to modify
array_to_modify = archive['arr_0']

# Print the original data type
print(f"Original Data Type: {array_to_modify.dtype}")

# Convert the array to int64 during modification
modified_array = array_to_modify.astype(np.int64)

# Print the modified data type
print(f"Modified Data Type: {modified_array.dtype}")

# Create a new dictionary with the modified array
new_archive = {key: value for key, value in archive.items()}
new_archive['arr_0'] = modified_array

# Save the modified archive to a new .npz file
new_file_path = 'train_data_2sats.npz'
np.savez(new_file_path, **new_archive)
