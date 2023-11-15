#pip install opencv-python
import matplotlib.pyplot as plt
import cv2

# Load the TIFF file using OpenCV
tif_image = cv2.imread('/Users/gbatu/OneDrive/Documents/DNB_VNP46A1_A2020225.tif', cv2.IMREAD_UNCHANGED)

# Normalize the pixel values to the range [0, 1] (assuming the original range is [0, 2])
tif_image_normalized = tif_image / 2.0

# Display the image
plt.imshow(tif_image_normalized, cmap="gray")  # Use a grayscale colormap
plt.colorbar()  # Add a color bar to indicate the values
plt.title("TIFF Image")
plt.show()