# %%
import random
import os
import fnmatch
import concurrent.futures
import traceback
from pathlib import Path

import numpy as np
import pydicom
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms

# %%
SEED = 6464
DATASET_DIR = '../../../../dataset'
RSNA_SRC_DIR = f'{DATASET_DIR}/rsna-pneumonia-detection-challenge'
RSNA_TARGET_DIR = f'{DATASET_DIR}/rsna-pneumonia-detection-challenge-jpg'
RSNA_RESIZED_DIR = f'{DATASET_DIR}/rsna-jpg-resized'
MAX_WORKERS = 16

# %%
random.seed(SEED)
np.random.seed(SEED)

# %%
def dicom_to_jpg(dcm_path_str, target_dir, pbar=None):
    try:
        dcm_id = Path(dcm_path_str).name.replace('.dcm', '')
        dcm = pydicom.dcmread(dcm_path_str)
        img = Image.fromarray(dcm.pixel_array)
        w, h = img.size
        dw, dh = (0, (w - h) // 2) if w >= h else ((h - w) // 2, 0)
        img = transforms.Pad((dw, dh))(img)
        img.save(f'{target_dir}/{dcm_id}.jpg', 'JPEG')
        if pbar is not None:
            pbar.update(1)
    except:
        print(f'Conversion failed for {dcm_path_str}')
        traceback.print_exc()


# %%
def convert_dicoms_to_jpg(src_dir, target_dir, total=None):
    dcm_gen = (os.path.join(base, filename)
               for base, dirs, files in os.walk(src_dir)
               for filename in fnmatch.filter(files, '*.dcm'))
    pbar = tqdm(total=total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for filename in dcm_gen:
        # for filename in list(dcm_gen)[:10]:
            executor.submit(dicom_to_jpg, filename, target_dir, pbar)
    pbar.close()

# %%
def convert_train_dicoms_to_jpg():
    train_src_dir = f'{RSNA_SRC_DIR}/stage_2_train_images'
    train_target_dir = f'{RSNA_TARGET_DIR}/stage_2_train_images'
    Path(train_target_dir).mkdir(parents=True, exist_ok=True)
    total = 26684
    convert_dicoms_to_jpg(train_src_dir, train_target_dir, total)

# %%
def convert_test_dicoms_to_jpg():
    test_src_dir = f'{RSNA_SRC_DIR}/stage_2_test_images'
    test_target_dir = f'{RSNA_TARGET_DIR}/stage_2_test_images'
    Path(test_target_dir).mkdir(parents=True, exist_ok=True)
    total = 3000
    convert_dicoms_to_jpg(test_src_dir, test_target_dir, total)

# %%
def convert_all_dicoms_to_jpg():
    convert_train_dicoms_to_jpg()

# %%
#convert_all_dicoms_to_jpg()

# %%
def resize(jpg_path_str, pbar=None):
    try:
        jpg_path = Path(jpg_path_str)
        img = Image.open(jpg_path_str)
        img = transforms.Resize((224, 224))(img)
        img.save(f'{RSNA_RESIZED_DIR}/stage_2_train_images/{jpg_path.name}')
        if pbar is not None:
            pbar.update(1)
    except:
        print(f'Resize failed for {jpg_path_str}')
        traceback.print_exc()

def resize_jpgs():
    Path(f'{RSNA_RESIZED_DIR}/stage_2_train_images').mkdir(parents=True, exist_ok=True)
    jpg_gen = (os.path.join(base, filename)
        for base, dirs, files in os.walk(f'{RSNA_TARGET_DIR}/stage_2_train_images')
        for filename in fnmatch.filter(files, '*.jpg'))
    pbar = tqdm(total=26684)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for filename in jpg_gen:
            executor.submit(resize, filename, pbar)
    pbar.close()

# %%
if __name__ == '__main__':
    resize_jpgs()
