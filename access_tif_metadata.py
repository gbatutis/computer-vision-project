# code from ChatGPT
import cv2
import numpy as np

# Specify the path to your TIFF image
image_path = '/Users/mallorysico/Desktop/Computer Vision/2021 Data Fusion Contest Data/dfc2021_dse_val_reference/1.tif'

# Load the TIFF image using OpenCV
tif_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if tif_image is not None:
    
    # Access pixel values at specific coordinates (x, y)
    x_ls = np.arange(16)  # Replace with your desired x-coordinate
    y_ls = np.arange(16)  # Replace with your desired y-coordinate

    for x in x_ls: 
        for y in y_ls: 
            # Extract pixel value at (x, y)
            pixel_value = tif_image[y, x]

            # Print the pixel value
            print("Pixel value at ({}, {}): {}".format(x, y, pixel_value))
else:
    print("Failed to load the image from '{}'".format(image_path))
