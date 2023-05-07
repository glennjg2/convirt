# %%
import random
import os
import fnmatch
import concurrent.futures
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms

# %%
SEED = 6464
DATASET_DIR = '../../../../dataset'
COVIDX_SRC_DIR = f'{DATASET_DIR}/covidx'
COVIDX_TARGET_DIR = f'{DATASET_DIR}/covidx-jpg'
MAX_WORKERS = 16

# %%
random.seed(SEED)
np.random.seed(SEED)

Path(f'{COVIDX_TARGET_DIR}/train').mkdir(parents=True, exist_ok=True)
Path(f'{COVIDX_TARGET_DIR}/test').mkdir(parents=True, exist_ok=True)

# %%
from tqdm.auto import tqdm

def convert_to_jpg(filename, mode='train', pbar=None):
    try:
        img_path = Path(f'{COVIDX_SRC_DIR}/{mode}/{filename}')
        if img_path.exists():
            img = Image.open(img_path)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            w, h = img.size
            dw, dh = (0, (w - h) // 2) if w >= h else ((h - w) // 2, 0)
            transform = transforms.Compose([transforms.Pad((dw, dh)), transforms.Resize((224, 224))])
            img = transform(img)
            img.save(f'{COVIDX_TARGET_DIR}/{mode}/{os.path.splitext(filename)[0]}.jpg', 'JPEG')
            if pbar is not None:
                pbar.update(1)
    except:
        print(f'Conversion failed for {filename}')
        traceback.print_exc()

def to_jpg_with_mode(mode, pbar=None):
    def row_to_jpg(row):
        convert_to_jpg(row['filename'], mode, pbar)
    return row_to_jpg

def convert():
    df = pd.read_csv(f'{COVIDX_SRC_DIR}/train.txt', sep=' ', header=None)
    df.columns = ['patient_id', 'filename', 'label', 'data_source']
    pbar = tqdm(total=29986)
    df.apply(to_jpg_with_mode(mode='train', pbar=pbar), axis=1)
    pbar.close()

    pbar = tqdm(total=4000)
    df = pd.read_csv(f'{COVIDX_SRC_DIR}/test.txt', sep=' ', header=None)
    df.columns = ['patient_id', 'filename', 'label', 'data_source']
    df.apply(to_jpg_with_mode(mode='test', pbar=pbar), axis=1)
    pbar.close()

# %%
if __name__ == '__main__':
    convert()
