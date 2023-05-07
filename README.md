# Introduction

This is a replication project for the paper, *"Contrastive Learning of Medical Visual Representations from Paired Images and Text"* by Zhang, et al., 2020.

# Environment

## Prerequisites

- Install Apache Maven: https://maven.apache.org/install.html
- Install conda: https://conda.io/projects/conda/en/stable/user-guide/install/index.html
- Physionet data access

## Set up conda environment

Run the following after installing `conda`, replacing `PREFERRED_LOCATION` with where you want to install python packages:

```
conda env create --file environment.yml --prefix <PREFERRED_LOCATION>
```

Activate your conda environment by running:

```
conda activate convirt
```

# Datasets

## Download MIMIC CXR JPG 2.0

Download the image files: https://physionet.org/content/mimic-cxr-jpg/2.0.0/.

Save or symlink to `dataset/mimic-cxr-jpg-2.0.0` so you will have:

```
dataset/
  └╴mimic-cxr-jpg-2.0.0
    ├╴p10
    ├╴p11
    ├╴...
    └╴p19
```

## Download MIMIC CXR Data

https://physionet.org/content/mimic-cxr/2.0.0/

- https://physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv.gz?download
- https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz?download
- https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip?download
  - Extract as `dataset/mimic-cxr-reports`
- https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz?download
  - Extract as `dataset/mimic-cxr-2.0.0-metadata.csv`
- https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz?download
  - Extract as `dataset/mimic-cxr-2.0.0-split.csv`

## Download ClinicalBERT

- Download from https://github.com/EmilyAlsentzer/clinicalBERT#download-clinical-bert
- Extract weights to `models/lib`
  - You should have the following structure
```
models/
  └╴lib
    └╴emilyalsentzer
      ├╴Bio_ClinicalBERT
      └╴ClinicalBERT
```

## Download RSNA

https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

Extract into `dataset/rsna-pneumonia-detection-challenge`.

## Download COVIDx

https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

Extract into `dataset/covidx`.

# Preprocessing

## Tokenization

At the root directory where `pom.xml` is, execute:

```
mvn compile exec:java
```

## Convert or Resize Images

From `src/main/python/preprocess` run:

```
python mimic.py

python rsna.py

python covidx.py
```

This should save the resized images into:

- MIMIC: `dataset/mimic-cxr-jpg-2.0.0-resized`
- RSNA: `dataset/rsna-pneumonia-detection-challenge-jpg/stage_2_train_images`
- COVIDx: `dataset/covidx-jpg/train` and `dataset/covidx-jpg/test`

# Training

## ConVIRT Contrastive Training

From `src/main/python` run:

```
python train.py -train 218000 -val 5000 -b 32 -stop 10 -f "convirt-best-{epoch:03d}-{val_auc:.4f}" convirt
```

Here is an explanation of the arguments:
- `-train 218000`, `-val 5000` limits the total number of training images to 218k, and validation images to 5k
- `-b 32` sets the batch size to 32
- `-stop 10` number of epochs for early stopping based on validation loss
- `-f "convirt-best-{epoch:03d}-{val_auc:.4f}"` specifies the checkpoint file naming scheme
- `convirt` means that we are doing convirt training

To load from a previous checkpoint and optionally specify a different learning rate, run:

```
python train.py -train 218000 -val 5000 -b 32 -stop 10 -f "convirt-best-{epoch:03d}-{val_auc:.4f}" --load-checkpoint "<CHECKPOINT_PATH>" -lr 2e-5 convirt
```
- `--load-checkpoint "<CHECKPOINT_PATH>"` points at the checkpoint file to be loaded
- `-lr 2e-5` specifies the learning rate to use

## ImageNet Training

To train ImageNet-based models for RSNA and COVIDx, run the corresponding commands below.

### RSNA

To train on a `FRACTION` portion of the RSNA dataset, run:
```
python train.py -fc 1024 -p 0.4 -lrp 4 -log 50 -stop 10 -train <FRACTION> -lr 2e-3 -f "<CHECKPOINT_SAVE_PATH>" rsna
```
- `-p 0.4` this specifies the dropout rate
- `-lrp 4` sets the learning rate scheduling check to every four epochs
- `-log 50` log results every 50 batches
- `-fc 1024` specifies that the linear layer should have size 1024
- `rsna` means we are running training on RSNA

For example, to train on 10% of the data:
```
python train.py -fc 1024 -p 0.4 -lrp 4 -log 50 -stop 10 -train 0.1  -lr 2e-3 -f "rsna-best-10pct" rsna
```

Leave out the `-train <FRACTION>` argument to train on 100% of the data.

### COVIDx

Training on COVIDx is similar. For example, to train on 10% of the data:
```
python train.py -log 50 -stop 10 -fc 1024 -train 0.1 -f "covidx-best-10pct" covidx
```

## CXR Contrastive Training

After the ConVIRT contrastive model has been trained, we can load the best checkpoint for use in downstream CXR classification tasks.

You can use the `-train <FRACTION>` argument to train on only a `FRACTION` portion of these datasets.

### RSNA

Load the best ConVIRT checkpoint and train on 100% of RSNA using this command:
```
python train.py -log 50 -stop 10 -fc 2048 --train-target convirt --convirt-checkpoint-path "<CONVIRT_BEST_CHECKPOINT_PATH>" -f "rsna-best-100pct-convirt" rsna
```

### COVIDx

The command is similar to that of RSNA, e.g., to train on 10% of COVIDx:
```
python train.py -log 50 -stop 10 -fc 2048 -train 0.1 --train-target convirt --convirt-checkpoint-path "<CONVIRT_BEST_CHECKPOINT_PATH>" -f "covidx-best-10pct-convirt" covidx
```