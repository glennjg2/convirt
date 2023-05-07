# %%
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import stanza
import lightning.pytorch as ling

from pathlib import Path
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomAffine,
    ColorJitter,
    GaussianBlur,
    Resize,
    ToTensor,
    Grayscale,
    Normalize
)
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from const import DATASET_DIR, MODELS_DIR, SEED


# %%
SubjectId = 'subject_id'
StudyId = 'study_id'
DicomId = 'dicom_id'
InputIds = 'input_ids'
TokenTypeIds = 'token_type_ids'
AttentionMask = 'attention_mask'
Labels = 'labels'

MIMIC_JPG_DIR = f'{DATASET_DIR}/mimic-cxr-jpg-2.0.0-resized'


# %%
def jpg_exists(row):
    path_str = f'{MIMIC_JPG_DIR}/p{str(row.subject_id)[:2]}/p{row.subject_id}/s{row.study_id}/{row.dicom_id}.jpg'
    return Path(path_str).exists()

def to_bin_to_dec(row):
    bin_str = ''.join(map(lambda v: str(int(v)), row.tolist()))
    return int(bin_str, 2)

def get_xr_labels_reports():
    df_labels = pd.read_csv(f'{DATASET_DIR}/metadata/mimic-cxr-2.0.0-chexpert.csv')
    df_labels = df_labels.replace(-1.0, 0.0)
    df_labels = df_labels.fillna(0)
    df_labels = df_labels.iloc[:, :-1]
    df_labels['multi_label'] = df_labels.iloc[:, 2:].apply(to_bin_to_dec, axis=1)
    df_labels['label'] = df_labels.multi_label.apply(lambda x: 1 if x > 0 else 0)
    df_labels = df_labels[[SubjectId, StudyId, 'label']]

    df_reports = pd.read_csv(f'{DATASET_DIR}/processed/mimic-cxr-reports/reports.csv')
    df_reports.report = df_reports.report.str.strip()
    df_reports = df_reports.sort_values([SubjectId, StudyId])
    df_reports.columns = [SubjectId, StudyId, 'sentence']

    df = pd.merge(df_reports, df_labels, on=[SubjectId, StudyId], how='inner')
    df = df[(df.sentence.str.split().apply(len) > 3) & (df.sentence != '')]

    return df

def get_xr_dicom_splits():
    df = pd.read_csv(f'{DATASET_DIR}/metadata/mimic-cxr-2.0.0-split.csv')
    df = df[[SubjectId, StudyId, DicomId, 'split']]
    return df

def get_xr_df():
    reports_df = get_xr_labels_reports()
    splits_df = get_xr_dicom_splits()
    merged_df = pd.merge(reports_df, splits_df, how='left', on=[SubjectId, StudyId])
    merged_df['found'] = merged_df.apply(jpg_exists, axis=1)
    merged_df = merged_df[merged_df.found]
    return merged_df.iloc[:, :-1].reset_index(drop=True)

def get_jpg_path(subject_id, study_id, dicom_id):
    return f'{MIMIC_JPG_DIR}/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'

def random_crop(img):
    w, h = img.size
    r = random.uniform(0.6, 1.0)
    w, h = int(w * r), int(h * r)
    return RandomCrop((h, w))


# %%
# (351894, 2835, 4637)
class ConvirtDataset(Dataset):

    def __init__(self, df, mode='train', subset=None):
        if mode not in {'train', 'validate', 'test', 'test_val', 'all'}:
            raise KeyError('mode')
        if mode == 'test_val':
            df = df[(df.split == 'test') | (df.split == 'validate')]
        elif mode != 'all':
            df = df[df.split == mode]
        if subset is not None and subset > 0:
            df = df.sample(n=int(subset)) if subset > 1 else df.sample(frac=subset)
        self.df = df[[SubjectId, StudyId, 'sentence', 'label', DicomId]].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(f'{MODELS_DIR}/lib/emilyalsentzer/ClinicalBERT')
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        jpg_path = get_jpg_path(row[SubjectId], row[StudyId], row[DicomId])
        report = row['sentence']
        label = row['label']

        img = Image.open(jpg_path)
        transform = Compose([
            Grayscale(num_output_channels=1),
            random_crop(img),
            RandomHorizontalFlip(p=0.5),
            RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
            GaussianBlur(kernel_size=(5,5), sigma=(0.1, 3.0)),
            Resize((224, 224)),
            ToTensor(),
            Normalize(0.449, 0.226)
        ])
        img = transform(img)

        text = random.sample(self.nlp(report).sentences, 1)[0].text
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=299, return_tensors='pt')

        return [img, tokenized, torch.tensor(label, dtype=torch.int)]

    @staticmethod
    def collate(batch):
        imgs, texts, labels = zip(*batch)

        input_ids = torch.stack([m[InputIds] for m in texts]).squeeze(1)
        token_type_ids = torch.stack([m[TokenTypeIds] for m in texts]).squeeze(1)
        attention_mask = torch.stack([m[AttentionMask] for m in texts]).squeeze(1)
        texts = {InputIds: input_ids, TokenTypeIds: token_type_ids, AttentionMask: attention_mask}

        return torch.stack(imgs), texts, torch.stack(labels)


# %%
class ConvirtDataModule(ling.LightningDataModule):

    def __init__(
        self,
        batch_size=64,
        shuffle_train=True,
        whole_batches=True,
        train_subset=None,
        val_subset=None,
        test_subset=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.whole_batches = whole_batches
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

    def setup(self, stage=None):
        xr_df = get_xr_df()
        self.train_dataset = ConvirtDataset(xr_df, subset=self.train_subset)
        self.val_dataset = ConvirtDataset(xr_df, mode='test_val', subset=self.val_subset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            drop_last=self.whole_batches,
            collate_fn=ConvirtDataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            drop_last=self.whole_batches,
            collate_fn=ConvirtDataset.collate
        )


# %%
class ConvirtModel(nn.Module):

    def __init__(self, encoding_size_d=512, tau=0.1, loss_weight=0.75):
        super().__init__()

        self.encoding_size_d = encoding_size_d
        self.tau = tau
        self.loss_weight = loss_weight

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=self.encoding_size_d)
        self.gv1 = nn.Linear(in_features=self.encoding_size_d, out_features=self.encoding_size_d)
        self.gv2 = nn.Linear(in_features=self.encoding_size_d, out_features=self.encoding_size_d)

        self.cbert = AutoModelForSequenceClassification.from_pretrained(f'{MODELS_DIR}/lib/emilyalsentzer/ClinicalBERT')
        for p in self.cbert.bert.embeddings.parameters():
            p.requires_grad = False
        for i in range(6):
            for p in self.cbert.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        self.cbert.classifier = nn.Linear(
            in_features=self.cbert.classifier.in_features,
            out_features=self.encoding_size_d
        )
        self.gu1 = nn.Linear(in_features=self.encoding_size_d, out_features=self.encoding_size_d)
        self.gu2 = nn.Linear(in_features=self.encoding_size_d, out_features=self.encoding_size_d)

    def forward(self, img, text):

        v = self.gv1(self.resnet(img))
        v = self.gv2(F.relu(v))

        u = self.gu1(self.cbert(**text).logits)
        u = self.gu2(F.relu(u))

        # reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html

        v_sim_u = F.cosine_similarity(v, u, dim=-1) / self.tau
        v_sim_all_u = F.cosine_similarity(v.unsqueeze(1), u.unsqueeze(0), dim=-1) / self.tau
        loss_vu = (-v_sim_u) + torch.logsumexp(v_sim_all_u, dim=-1)

        u_sim_v = F.cosine_similarity(u, v, dim=-1) / self.tau
        u_sim_all_v = F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=-1) / self.tau
        loss_uv = (-u_sim_v) + torch.logsumexp(u_sim_all_v, dim=-1)

        L = (self.loss_weight * loss_vu + (1 - self.loss_weight) * loss_uv).mean()

        return L

    def encode(self, img):
        return self.resnet(img)


# %%
class ConvirtLingModule(ling.LightningModule):

    def __init__(self, convirt, lr=1e-4, weight_decay=1e-6, lr_factor=0.5, lr_patience=5) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['convirt'])
        self.convirt = convirt
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

    def training_step(self, batch, batch_idx):
        img, text, labels = batch
        loss = self.convirt(img, text)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        img, text, labels = batch
        loss = self.convirt(img, text)
        self.log('val_loss', loss.item())

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        reducer = ReduceLROnPlateau(
            adam, mode='min', factor=self.lr_factor, patience=self.lr_patience, verbose=True
        )
        return {
            'optimizer': adam,
            'lr_scheduler': {
                'scheduler': reducer,
                'monitor': 'val_loss',
            }
        }


# %%
def get_callbacks(args):
    checkpt_best = ModelCheckpoint(
        dirpath=f'{MODELS_DIR}/checkpoints/',
        save_top_k=args.save_top_k,
        monitor='val_loss',
        mode='min',
        filename=args.checkpoint_filename
    )
    checkpt_analysis = ModelCheckpoint(
        dirpath=f'{MODELS_DIR}/checkpoints/',
        save_top_k=-1,
        every_n_epochs=5,
        monitor='val_loss',
        mode='min',
        filename=f'analysis-{args.checkpoint_filename}'
    )
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=args.early_stop_patience)
    return [checkpt_best, checkpt_analysis, early_stop]

def get_data_module(args):
    return ConvirtDataModule(
        batch_size=args.batch_size,
        shuffle_train=args.shuffle_train,
        whole_batches=args.whole_batches,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )

def get_model_module(model, args):
    return ConvirtLingModule(
        model,
        lr=args.learning_rate,
        lr_factor=args.lr_factor,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
    )

def get_convirt_resnet(args):
    convirt = ConvirtModel(encoding_size_d=args.encoding_size, tau=args.tau, loss_weight=args.loss_w_lambda)
    module = ConvirtLingModule.load_from_checkpoint(args.convirt_checkpoint_path, convirt=convirt)
    return module.convirt.resnet