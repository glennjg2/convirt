# %%
import os
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as ling

from torchvision.models import resnet50, ResNet50_Weights
from lightning.pytorch import seed_everything

import convirt
import covidx
import rsna
import args as A

from convirt import get_convirt_resnet


# %%
def get_trainer(args, callbacks):
    return ling.Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        callbacks=callbacks
    )


def run(args, model_module, data_module, callbacks):
    seed_everything(args.seed, workers=True)
    trainer = get_trainer(args, callbacks)
    trainer.fit(model=model_module, datamodule=data_module)
    if not args.fast_dev_run:
        trainer.test(ckpt_path='best', datamodule=data_module)


# %%
if __name__ == '__main__':
    args = A.parse_args()
    if args.model is None:
        raise ValueError('model is required')
    seed_everything(args.seed, workers=True)
    if args.model == 'covidx' or args.model == 'rsna':
        use_convirt = args.train_target == 'convirt'
        if args.model == 'covidx':
            src = covidx
            construct_model = covidx.CovidxModel
        else:
            src = rsna
            construct_model = rsna.RsnaModel
        resnet = get_convirt_resnet(args) if use_convirt else resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = construct_model(**{
            'resnet': resnet,
            'p': args.dropout_p,
            'fc_features': args.fc_features,
            'convirt': use_convirt
        })
        run(args, src.get_model_module(model, args), src.get_data_module(args), src.get_callbacks(args))
    elif args.model == 'convirt':
        model = convirt.ConvirtModel(encoding_size_d=args.encoding_size, tau=args.tau, loss_weight=args.loss_w_lambda)
        run(args, convirt.get_model_module(model, args), convirt.get_data_module(args), convirt.get_callbacks(args))
    else:
        print('Nothing to do.')