# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import lightning.pytorch as ling

from torchvision.transforms import ToTensor, Compose, Normalize, Grayscale
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from const import DATASET_DIR, MODELS_DIR, SEED

# %%
class RsnaDataset(Dataset):

    JPG_DIR = f'{DATASET_DIR}/rsna-jpg-resized/stage_2_train_images'

    def __init__(self, df, subset=None):
        if subset is not None and subset > 0:
            df = df.sample(n=int(subset)) if subset > 1 else df.sample(frac=subset)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patientId']
        label = row['Target']
        img = Image.open(f'{RsnaDataset.JPG_DIR}/{patient_id}.jpg')
        transform = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(0.449, 0.226)
        ])
        img = transform(img)
        return img, torch.tensor(label, dtype=torch.float)

    @staticmethod
    def get_rsna_df():
        df = pd.read_csv(f'{DATASET_DIR}/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
        df_pt = df[['patientId', 'Target']]
        dupes_idx = df_pt[df_pt.duplicated()].index
        df = df[~df.index.isin(dupes_idx)].reset_index(drop=True)
        return df


# %%
class RsnaLingDataModule(ling.LightningDataModule):

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
        df = RsnaDataset.get_rsna_df()
        train_idxs, test_idxs = train_test_split(df.index, test_size=0.1124, random_state=SEED)
        train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.0633, random_state=SEED)

        train_df = df.iloc[train_idxs].reset_index(drop=True)
        val_df = df.iloc[val_idxs].reset_index(drop=True)
        test_df = df.iloc[test_idxs].reset_index(drop=True)

        self.train_dataset = RsnaDataset(train_df, subset=self.train_subset)
        self.val_dataset = RsnaDataset(val_df, subset=self.val_subset)
        self.test_dataset = RsnaDataset(test_df, subset=self.test_subset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train, drop_last=self.whole_batches)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=self.whole_batches)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=self.whole_batches)


# %%
class RsnaModel(nn.Module):

    def __init__(self, resnet, p=0.2, fc_features=1024, encoding_d=512, convirt=False):
        super().__init__()
        self.p = p
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        if not convirt:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if fc_features > 1:
            self.resnet.fc = nn.Sequential(
                nn.Linear(in_features=self.resnet.fc.in_features, out_features=encoding_d),
                nn.ReLU(),
                nn.Linear(in_features=encoding_d, out_features=fc_features),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(in_features=fc_features, out_features=1)
            )
        else:
            self.resnet.fc = nn.Sequential(
                nn.Linear(in_features=self.resnet.fc.in_features, out_features=encoding_d),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(in_features=encoding_d, out_features=1)
            )

    def forward(self, img):
        return self.resnet(img)


# %%
class RsnaLinearLingModule(ling.LightningModule):

    def __init__(self, rsna, lr=1e-4, weight_decay=1e-6, lr_factor=0.5, lr_patience=5) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['rsna'])
        self.rsna = rsna
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

    def training_step(self, batch, batch_idx):
        img, y_true = batch
        y = self.rsna(img).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(y, y_true)
        auc = roc_auc_score(y_true.detach().float().cpu().numpy(), F.sigmoid(y).detach().float().cpu().numpy())
        self.log_dict({'train_auc': auc, 'train_loss': loss.item()})
        return loss

    def evaluate(self, batch, stage=None):
        img, y_true = batch
        y = self.rsna(img).squeeze(-1)
        auc = roc_auc_score(y_true.detach().float().cpu().numpy(), F.sigmoid(y).detach().float().cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(y, y_true)
        if stage:
            self.log_dict({f'{stage}_auc': auc, f'{stage}_loss': loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

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
        monitor='val_auc',
        mode='max',
        filename=args.checkpoint_filename
    )
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=args.early_stop_patience)
    return [checkpt_best, early_stop]


def get_data_module(args):
    return RsnaLingDataModule(
        batch_size=args.batch_size,
        shuffle_train=args.shuffle_train,
        whole_batches=args.whole_batches,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )


def get_model_module(model, args):
    return RsnaLinearLingModule(
        model,
        lr=args.learning_rate,
        lr_factor=args.lr_factor,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
    )