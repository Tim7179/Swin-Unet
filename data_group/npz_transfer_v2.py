from PIL import Image
import numpy as np
import os
from os.path import join as Path_Join
from tqdm import tqdm
import shutil

GROUP_FOLDER = 'new_2023-12-10'
GROUP_LEN = 10
TEST_GROUPS = [0]  # Start from 0
MAIN_CODE_FOLDER = '../Swin-Unet/'
TRAIN_FOLDER = '../Swin-Unet/data/Synapse/train_npz/'
TEST_FOLDER = '../Swin-Unet/data/Synapse/test_vol_h5/'
LIST_FOLDER = '../Swin-Unet/lists/lists_Synapse/'

IMAGE_SIZE = 224

# Clear or create the target folder
def initial_folder(folder):
    # Clear folder if exists
    if os.path.exists(folder):
        print(f"Clear folder: {folder}")
        shutil.rmtree(folder)

    print(f"Create folder: {folder}")
    os.makedirs(folder)


def npz_transfer(images_path: str, labels_path: str, target_path: str, is_test_data=False, title=""):

    image_names = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    image_names.sort()

    # Transfer images and labels to .npz file
    for image_name in tqdm(image_names, desc=title):
        image_path = Path_Join(images_path, image_name)
        label_path = Path_Join(labels_path, image_name)

        # Read image
        image = np.array(Image.open(image_path).convert('RGB'))

        # Read label and convert image to binary
        
        label_binary = np.array(Image.open(label_path).convert('L').point())
        
        print(label_binary.shape)
        # Resize images if not match target
        if image.shape[0] != IMAGE_SIZE:
            print(f"\rresize {image_name}: from {image.shape[0]} to {IMAGE_SIZE}")
            image = np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE)))
            label_binary = np.array(Image.fromarray(label_binary).resize((IMAGE_SIZE, IMAGE_SIZE)))

        if is_test_data:
            image = image.transpose(2, 0, 1)
        else:
            image = np.array(Image.fromarray(image).convert('L'))
            image = np.clip(image, 0, 255)

        # Save image and label to .npz file
        np.savez(Path_Join(target_path, image_name[:-4] + ".npz"), image=image, label=label_binary, case=image_name[:-4])

def make_list(data_path, list_path, title=""):
    file_names = [f for f in os.listdir(data_path) if f.lower().endswith('.npz')]
    with open(list_path, 'w') as f:
        for file_name in tqdm(file_names, desc=title):
            f.write(file_name[:-4] + '\n')


if __name__ == "__main__":
    TRAIN_GROUPS = [i for i in range(GROUP_LEN) if i not in TEST_GROUPS]
    print("Train_groups:", TRAIN_GROUPS)
    print("Test_groups:", TEST_GROUPS)

    # Clear or create the target folder
    print("Clear or create the target folder")
    initial_folder(TRAIN_FOLDER)
    initial_folder(TEST_FOLDER)

    # Transfer test_data
    for i in TEST_GROUPS:
        npz_transfer(Path_Join(GROUP_FOLDER, str(i), "imgs"), Path_Join(GROUP_FOLDER, str(i), "masks"),
                     TEST_FOLDER, is_test_data=True, title=f"Group {i}")

    # Transfer train_data
    for i in TRAIN_GROUPS:
        npz_transfer(Path_Join(GROUP_FOLDER, str(i), "imgs"), Path_Join(GROUP_FOLDER, str(i), "masks"),
                     TRAIN_FOLDER, is_test_data=False, title=f"Group {i}")

    # Make List
    print("Make List")
    # Make list file for train_data
    make_list(TRAIN_FOLDER, Path_Join(LIST_FOLDER, 'train.txt'), title="Train")
    # Make list file for test_data
    make_list(TEST_FOLDER, Path_Join(LIST_FOLDER, 'test_vol.txt'), title="Test")
