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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from const import DATASET_DIR, MODELS_DIR


# %%
class CovidxDataset(Dataset):

    COVIDX_SRC_DIR = f'{DATASET_DIR}/covidx'
    COVIDX_JPG_DIR = f'{DATASET_DIR}/covidx-jpg'
    VALIDATION_SPLIT = 0.02158584

    def __init__(self, df, mode='train', subset=None):
        if subset is not None and subset > 0:
            df = df.sample(n=int(subset)) if subset > 1 else df.sample(frac=subset)
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['label']
        label = 1 if label == 'positive' else 0
        img = Image.open(f'{CovidxDataset.COVIDX_JPG_DIR}/{self.mode}/{os.path.splitext(filename)[0]}.jpg')
        transform = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(0.449, 0.226)
        ])
        img = transform(img)
        return img, torch.tensor(label, dtype=torch.float)

    @staticmethod
    def get_train_val_df():
        df = pd.read_csv(f'{CovidxDataset.COVIDX_SRC_DIR}/train.txt', sep=' ', header=None)
        df.columns = ['patient_id', 'filename', 'label', 'data_source']
        val_df = df.sample(frac=CovidxDataset.VALIDATION_SPLIT)
        train_df = df[~df.index.isin(val_df.index)].reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        return train_df, val_df

    @staticmethod
    def get_test_df():
        test_df = pd.read_csv(f'{CovidxDataset.COVIDX_SRC_DIR}/test.txt', sep=' ', header=None)
        test_df.columns = ['patient_id', 'filename', 'label', 'data_source']
        test_df = test_df.reset_index(drop=True)
        return test_df


# %%
class CovidxLingDataModule(ling.LightningDataModule):

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
        train_df, val_df = CovidxDataset.get_train_val_df()
        test_df = CovidxDataset.get_test_df()
        self.train_dataset = CovidxDataset(train_df, subset=self.train_subset)
        self.val_dataset = CovidxDataset(val_df, subset=self.val_subset)
        self.test_dataset = CovidxDataset(test_df, mode='test', subset=self.test_subset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train, drop_last=self.whole_batches
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=self.whole_batches)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=self.whole_batches)


# %%
class CovidxModel(nn.Module):

    def __init__(self, resnet, p=0.2, fc_features=1024, encoding_d=512, convirt=False):
        super().__init__()
        self.p = p
        self.resnet = resnet
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
class CovidxLingModule(ling.LightningModule):

    def __init__(self, covidx, lr=1e-4, weight_decay=1e-6, lr_factor=0.5, lr_patience=5) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['covidx'])
        self.covidx = covidx
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

    def training_step(self, batch, batch_idx):
        img, y_true = batch
        y = self.covidx(img).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(y, y_true)
        self.log('train_loss', loss.item())
        return loss

    def evaluate(self, batch, stage=None):
        img, y_true = batch
        y = self.covidx(img).squeeze(-1)
        y_pred = (F.sigmoid(y) > 0.5).int()
        y_true_, y_pred_ = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        acc = accuracy_score(y_true_, y_pred_)
        f1 = f1_score(y_true_, y_pred_, zero_division=1)
        loss = F.binary_cross_entropy_with_logits(y, y_true)
        if stage:
            self.log_dict({f'{stage}_loss': loss, f'{stage}_accuracy': acc, f'{stage}_f1': f1}, prog_bar=True)

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
        monitor='val_accuracy',
        mode='max',
        filename=args.checkpoint_filename
    )
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=args.early_stop_patience)
    return [checkpt_best, early_stop]


def get_data_module(args):
    return CovidxLingDataModule(
        batch_size=args.batch_size,
        shuffle_train=args.shuffle_train,
        whole_batches=args.whole_batches,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )


def get_model_module(model, args):
    return CovidxLingModule(
        model,
        lr=args.learning_rate,
        lr_factor=args.lr_factor,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
    )
