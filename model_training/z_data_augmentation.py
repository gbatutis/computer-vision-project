import torch
import numpy as np
from torchvision import transforms

loaded_data = np.load("/scratch/gsb8013/cv-final-project/model_training/train_electricity.npz")
input_tensor = loaded_data["arr_0"]

input_tensor = torch.from_numpy(input_tensor)

# Split the tensor along the second dimension (dim=1)
split_tensors = torch.chunk(input_tensor, chunks=57, dim=1)

# Check the shape of the first split tensor
print("split_tensors",split_tensors[0].shape)
print(len(split_tensors))

squeezed_tensor =[]
for i in range(len(split_tensors)):
    squeezed_tensor.append(split_tensors[i].squeeze(1))
print("squeezed_tensor",squeezed_tensor[0].shape)
print(len(squeezed_tensor))

# Define the transformations
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0,360)),
    transforms.RandomResizedCrop(size=(50,50))
])

augmented_tensors = []
# Apply transformations to each tensor in the list
for i in range(len(squeezed_tensor)):
    x= (squeezed_tensor[i]) #data_transform
    x = torch.unsqueeze(x, 1)
    augmented_tensors.append(x)
print("augmented_tensors",augmented_tensors[0].shape) 
#augmented_tensors = [data_transform(tensor) for tensor in input_tensor]
print("augmented_tensors",len(augmented_tensors))

# Convert the list of tensors back to a single tensor
#augmented_tensor = torch.stack(augmented_tensors)
concatenated_tensor = torch.cat(augmented_tensors, dim=1)

# Check the shape of the augmented tensor
print("concatenated_tensor",concatenated_tensor.shape)

# Convert augmented_data back to numpy array
arr_0 = concatenated_tensor.numpy()

# Save the augmented data to an npz file
np.savez("train_electricity_aug2.npz", arr_0)
print("Saved")
