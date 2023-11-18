import cv2
import numpy as np

# Load the image
# image_path = '/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/dfc2021_dse_val/Val/Tile19/S1A_IW_GRDH_20200828_VV.tif'
# image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

channels = [
    'DNB_VNP46A1_A2020221',
    'DNB_VNP46A1_A2020224',
    'DNB_VNP46A1_A2020225',
    'DNB_VNP46A1_A2020226',
    'DNB_VNP46A1_A2020227',
    'DNB_VNP46A1_A2020231',
    'DNB_VNP46A1_A2020235',
    'DNB_VNP46A1_A2020236',
    'DNB_VNP46A1_A2020237',
    'L2A_20200811_B01',
    'L2A_20200811_B02',
    'L2A_20200811_B03',
    'L2A_20200811_B04',
    'L2A_20200811_B05',
    'L2A_20200811_B06',
    'L2A_20200811_B07',
    'L2A_20200811_B08',
    'L2A_20200811_B09',
    'L2A_20200811_B11',
    'L2A_20200811_B12',
    'L2A_20200811_B8A',
    'L2A_20200816_B01',
    'L2A_20200816_B02',
    'L2A_20200816_B03',
    'L2A_20200816_B04',
    'L2A_20200816_B05',
    'L2A_20200816_B06',
    'L2A_20200816_B07',
    'L2A_20200816_B08',
    'L2A_20200816_B09',
    'L2A_20200816_B11',
    'L2A_20200816_B12',
    'L2A_20200816_B8A',
    'L2A_20200826_B01',
    'L2A_20200826_B02',
    'L2A_20200826_B03',
    'L2A_20200826_B04',
    'L2A_20200826_B05',
    'L2A_20200826_B06',
    'L2A_20200826_B07',
    'L2A_20200826_B08',
    'L2A_20200826_B09',
    'L2A_20200826_B11',
    'L2A_20200826_B12',
    'L2A_20200826_B8A',
    'L2A_20200831_B01',
    'L2A_20200831_B02',
    'L2A_20200831_B03',
    'L2A_20200831_B04',
    'L2A_20200831_B05',
    'L2A_20200831_B06',
    'L2A_20200831_B07',
    'L2A_20200831_B08',
    'L2A_20200831_B09',
    'L2A_20200831_B11',
    'L2A_20200831_B12',
    'L2A_20200831_B8A'#,
    # 'LC08_L1TP_2020-07-29_B1',
    # 'LC08_L1TP_2020-07-29_B10',
    # 'LC08_L1TP_2020-07-29_B11',
    # 'LC08_L1TP_2020-07-29_B2',
    # 'LC08_L1TP_2020-07-29_B3',
    # 'LC08_L1TP_2020-07-29_B4',
    # 'LC08_L1TP_2020-07-29_B5',
    # 'LC08_L1TP_2020-07-29_B6',
    # 'LC08_L1TP_2020-07-29_B7',
    # 'LC08_L1TP_2020-07-29_B8',
    # 'LC08_L1TP_2020-07-29_B9',
    # 'LC08_L1TP_2020-08-14_B1',
    # 'LC08_L1TP_2020-08-14_B10',
    # 'LC08_L1TP_2020-08-14_B11',
    # 'LC08_L1TP_2020-08-14_B2',
    # 'LC08_L1TP_2020-08-14_B3',
    # 'LC08_L1TP_2020-08-14_B4',
    # 'LC08_L1TP_2020-08-14_B5',
    # 'LC08_L1TP_2020-08-14_B6',
    # 'LC08_L1TP_2020-08-14_B7',
    # 'LC08_L1TP_2020-08-14_B8',
    # 'LC08_L1TP_2020-08-14_B9',
    # 'LC08_L1TP_2020-08-30_B1',
    # 'LC08_L1TP_2020-08-30_B10',
    # 'LC08_L1TP_2020-08-30_B11',
    # 'LC08_L1TP_2020-08-30_B2',
    # 'LC08_L1TP_2020-08-30_B3',
    # 'LC08_L1TP_2020-08-30_B4',
    # 'LC08_L1TP_2020-08-30_B5',
    # 'LC08_L1TP_2020-08-30_B6',
    # 'LC08_L1TP_2020-08-30_B7',
    # 'LC08_L1TP_2020-08-30_B8',
    # 'LC08_L1TP_2020-08-30_B9',
    # 'S1A_IW_GRDH_20200723_VH',
    # 'S1A_IW_GRDH_20200723_VV',
    # 'S1A_IW_GRDH_20200804_VH',
    # 'S1A_IW_GRDH_20200804_VV',
    # 'S1A_IW_GRDH_20200816_VH',
    # 'S1A_IW_GRDH_20200816_VV',
    # 'S1A_IW_GRDH_20200828_VH',
    # 'S1A_IW_GRDH_20200828_VV'
]

# Define block size and calculate the number of blocks
block_size = 50
num_blocks = 256
blocks_per_row = 800 // block_size
blocks_per_col = 800 // block_size

# Initialize a list to store the smaller tensors
tensor_list = []

for i in range(1,20):
    small_tensors = []
    for j in channels:
        image_path = '/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/dfc2021_dse_val/Val/Tile{}/{}.tif'.format(i,j)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #print("image size", image.shape)
        if image is not None:

            # Initialize a list to store blocks for each image
            blocks_for_image = []

            # Iterate to cut the image into smaller blocks
            for row in range(blocks_per_row):
                for col in range(blocks_per_col):
                    # Extract a 50x50 pixel block
                    block = image[row * block_size: (row + 1) * block_size, col * block_size: (col + 1) * block_size]
                    blocks_for_image.append(block)

            # Stack blocks for each image along the first axis
            stacked_blocks_for_image = np.stack(blocks_for_image, axis=0)
            #print("stacked_blocks_for_image",stacked_blocks_for_image.shape)
            small_tensors.append(stacked_blocks_for_image)
            #print("small_tensors",len(small_tensors))

        # Stack the small tensors along the second axis
    final_stacked_tensor = np.stack(small_tensors, axis=0)
    final_stacked_tensor = np.transpose(final_stacked_tensor, (1, 0, 2, 3))
    print("final_stacked_tensor size", final_stacked_tensor.shape)
    tensor_list.append(final_stacked_tensor)

stacked_tensor = np.concatenate(tensor_list, axis=0)
print("stacked_tensor size:", stacked_tensor.shape)
np.savez_compressed(f'/Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/val_data_2sats.npz', stacked_tensor)
#for idx, array in enumerate(tensor_list):
#    np.savez_compressed('Users/gbatu/OneDrive/Documents/NYU/Semester_3/CV/val_file_{}.npz'.format(idx), data=array)