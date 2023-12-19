import numpy as np
loaded_data = np.load("/scratch/gsb8013/cv-final-project/model_training/val_labels2.npz")
import_tensor = loaded_data["labels"]


print("0",np.count_nonzero(import_tensor == 0))
print("1",np.count_nonzero(import_tensor == 1))
print("2",np.count_nonzero(import_tensor == 2))
print("3",np.count_nonzero(import_tensor == 3))
print("4",np.count_nonzero(import_tensor == 4))
