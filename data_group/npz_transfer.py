import cv2
import numpy as np
import os
from os.path import join as Path_Join
from tqdm import tqdm
import shutil

GROUP_FOLDER = '.'
GROUP_LEN = 10
TEST_GROUPS = [0] # Start from 0
MAIN_CODE_FOLDER = '../Swin-Unet/'
TRAIN_FOLDER = '../Swin-Unet/data/Synapse/train_npz/'
TEST_FOLDER = '../Swin-Unet/data/Synapse/test_vol_h5/'
LIST_FOLDER = '../Swin-Unet/lists/lists_Synapse/'

IMAGE_SIZE = 224
THRESHOLD = 128

# Clear or create the target folder
def inital_folder(folder):
    # Clear folder if exists
    if os.path.exists(folder):
        print(f"Clear folder: {folder}")
        shutil.rmtree(folder)
    
    print(f"Create folder: {folder}")
    os.makedirs(folder)

def convert_to_binary(image):
    # limit pixel value in [0, 255]
    image = np.clip(image, 0, 255)

    # use threshold to convert to binary
    _, binary_image = cv2.threshold(image, THRESHOLD, 1, cv2.THRESH_BINARY)

    return binary_image

def npz_transfer(images_path:str, labels_path:str, target_path:str, is_test_data=False, title=""):

    image_names = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    image_names.sort()

    # Transfer images and labels to .npz file
    for image_name in tqdm(image_names,desc=title):
        image_path = Path_Join(images_path, image_name)
        label_path = Path_Join(labels_path, image_name)

        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_binary = convert_to_binary(label)
        
        # Resize images if not match target
        if image.shape[0] != IMAGE_SIZE or label_binary.shape[0] != IMAGE_SIZE:
            print(f"\rresize {image_name}: from {image.shape[0]} to {IMAGE_SIZE}")
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            label_binary = cv2.resize(label_binary, (IMAGE_SIZE, IMAGE_SIZE))

        if is_test_data:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = image.transpose(2,0,1)

        # save image and label to .npz file 
        np.savez(Path_Join(target_path,image_name[:-4]+".npz"),image=image,label=label_binary)

def make_list(data_path, list_path, title=""):
    file_names = [f for f in os.listdir(data_path) if f.lower().endswith('.npz')]
    with open(list_path,'w') as f:
        for file_name in tqdm(file_names,desc=title):
            f.write(file_name[:-4]+'\n')
    

if __name__ == "__main__":
    TRAIN_GROUPS = [i for i in range(GROUP_LEN) if i not in TEST_GROUPS]
    print("Train_groups:",TRAIN_GROUPS)
    print("Test_groups:",TEST_GROUPS)

    # Clear or create the target folder
    print("Clear or create the target folder")
    inital_folder(TRAIN_FOLDER)
    inital_folder(TEST_FOLDER)
    

    # Transfer test_data
    for i in TEST_GROUPS:
        npz_transfer(Path_Join(GROUP_FOLDER, str(i),"imgs"),Path_Join(GROUP_FOLDER,str(i),"masks"),
                     TEST_FOLDER,is_test_data=False, title=f"Group {i}")

    # Transfer train_data
    for i in TRAIN_GROUPS:
        npz_transfer(Path_Join(GROUP_FOLDER, str(i),"imgs"),Path_Join(GROUP_FOLDER,str(i),"masks"),
                     TRAIN_FOLDER,is_test_data=True, title=f"Group {i}")
    
    # Make List
    print("Make List")
    # Make list file for train_data
    make_list(TRAIN_FOLDER, Path_Join(LIST_FOLDER,'train.txt'), title="Train")
    # Make list file for test_data
    make_list(TEST_FOLDER, Path_Join(LIST_FOLDER,'test_vol.txt'), title="Test")