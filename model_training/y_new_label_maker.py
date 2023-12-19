import numpy as np

loaded_data = np.load("/scratch/gsb8013/cv-final-project/model_training/val_labels2.npz")
input_tensor = loaded_data["labels"]

def custom_logic(number):
    # Example logic: if the number is 2, make it 0
    if number == 0:
        return 1
    if number == 1:
        return 0
    if number == 2:
        return 1
    if number == 3:
        return 0
    # Add more conditions as needed

    # If no specific condition is met, return the original number
    return number

# Vectorize the custom_logic function to apply it element-wise
vectorized_logic = np.vectorize(custom_logic)

# Apply the logic to each element in the array
new_numbers = vectorized_logic(input_tensor)

print("new_numbers len", len(new_numbers))
print("min, max", min(new_numbers)," ", max(new_numbers))
print("count of 1", np.count_nonzero(new_numbers ==1))
print("type", type(new_numbers[0]))
# Convert augmented_data back to numpy array
labels = new_numbers

# Save the augmented data to an npz file
np.savez("val_labels_humans.npz", labels)
