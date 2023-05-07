# %%
import os
import fnmatch
import concurrent.futures
import traceback
import random
import time
import threading

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from PIL import Image


# %%
DATASET_DIR = '../../../../dataset'
MIMIC_JPG_DIR = f'{DATASET_DIR}/mimic-cxr-jpg-2.0.0'
MIMIC_JPG_RESIZE_OUTPUT_DIR = f'{DATASET_DIR}/mimic-cxr-jpg-2.0.0-resized'

# %%
def resize(jpg_path, pbar=None):
    img = Image.open(jpg_path)
    out_path = jpg_path.replace(MIMIC_JPG_DIR, MIMIC_JPG_RESIZE_OUTPUT_DIR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = img.size
    sw, sh = (256 * h) // w, (256 * w) // h
    to_size = (256, sw) if w >= h else (sh, 256)
    img_ = transforms.Compose([transforms.Resize((to_size[1], to_size[0]))])(img)
    img_.save(out_path, 'JPEG')
    if pbar is not None:
        pbar.update(1)


# %%
MaxWorkers = 16

def resize_all_images():
    basedir = f'{MIMIC_JPG_DIR}'
    jpg_gen = (os.path.join(root, filename)
       for root, dirs, files in os.walk(basedir)
       for filename in fnmatch.filter(files, '*.jpg'))
    pbar = tqdm(total=377024, ncols=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MaxWorkers) as executor:
        for filename in jpg_gen:
            executor.submit(resize, filename, pbar)
    pbar.close()


# %%
if __name__ == '__main__':
    resize_all_images()
