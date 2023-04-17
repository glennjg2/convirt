
# Prerequisites

- Maven
- Python 3.8+
- PyTorch
- Pandas
- transformers (huggingface)


# Download MIMIC CXR JPG 2.0

Download the image files: https://physionet.org/content/mimic-cxr-jpg/2.0.0/.

Save to `dataset/mimic-cxr-jpg-2.0.0` so you will have:

```
dataset/
  └╴mimic-cxr-jpg-2.0.0
    ├╴p10
    ├╴p11
    ├╴...
    └╴p19
```

I made a symlink from `dataset/mimic-cxr-jpg-2.0.0` to the actual location in my storage device. Same for the preprocessed resized jpgs in `dataset/mimic-cxr-jpg-2.0.0-resized`.

# Download MIMIC CXR Data

https://physionet.org/content/mimic-cxr/2.0.0/

- https://physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv.gz?download
- https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz?download
- https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip?download
  - Extract as `dataset/mimic-cxr-reports`
- https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz?download
  - Extract as `dataset/mimic-cxr-2.0.0-metadata.csv`
- https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz?download
  - Extract as `dataset/mimic-cxr-2.0.0-split.csv`

# Download ClinicalBERT

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

# Run Tokenizer.java

At the root directory where `pom.xml` is, execute:

`mvn compile exec:java`

# Resize Images

Run `resize_all_images()` from the notebook `preprocess.ipynb`

This should save the resized images into `dataset/mimic-cxr-jpg-2.0.0-resized`.

# ConVIRT Pretraining

In the notebook `train_convirt.ipynb`:
- Create a training config, e.g., `config = TrainingConfig(train_sub=218000, val_sub=5000)`
- Run training by passing the config into `train()`, e.g., `train(config, batch_log_idx=100)`
