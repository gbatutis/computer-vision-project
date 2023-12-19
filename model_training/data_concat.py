import numpy as np

def concatenate_and_save_npz(file_paths, output_npz_path):
    # Load arrays from each NPZ file
    arrays = [np.load(file_path)['arr_0'] for file_path in file_paths]

    # Concatenate arrays along axis=0
    concatenated_array = np.concatenate(arrays, axis=0)

    # Save the concatenated array to a new NPZ file
    np.savez(output_npz_path, arr_0=concatenated_array)

    print(f"Concatenated array saved to: {output_npz_path}")
    print("array shape: ",concatenated_array.shape)

# Example usage:
npz_file_paths = [
    '/scratch/gsb8013/cv-final-project/model_training/train_data_1.npz',
    '/scratch/gsb8013/cv-final-project/model_training/train_data_2.npz',
    '/scratch/gsb8013/cv-final-project/model_training/train_data_3.npz'
]

output_npz_path = '/scratch/gsb8013/cv-final-project/model_training/train_data_final.npz'

concatenate_and_save_npz(npz_file_paths, output_npz_path)

