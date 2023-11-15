import os

def does_not_contain_ground_truth(path):
    return "groundTruth" not in path

# Specify the folder path
for i in range(1,20):
    folder_path = f"/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/dfc2021_dse_val/Val/Tile{i}"

    # List all files in the folder
    files = os.listdir(folder_path)

    # Generate Unix-style file paths for each file
    unix_file_paths = [os.path.join(folder_path, file) for file in files]

    # Print the generated file paths
    for path in unix_file_paths:
        if does_not_contain_ground_truth(path):
            print("cv2.imread('"+path+"', cv2.IMREAD_UNCHANGED),")

#see where you are
#print(os.getcwd())