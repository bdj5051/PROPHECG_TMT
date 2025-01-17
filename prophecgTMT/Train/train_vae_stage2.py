
from ecgxaiY.systems.VAE_system import GaussianVAE
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import MetricCollection, MeanSquaredError

import yaml
import pandas as pd
import os

from ecgxaiY.utils.dataset import UniversalECGDataset
from ecgxaiY.utils.transforms import ApplyGain, ToTensor, To12Lead, Resample
from ecgxaiY.utils.metrics import TMW
from ecgxaiY.network.causalcnn.encoder import CausalCNNVEncoder
from ecgxaiY.network.causalcnn.decoder import CausalCNNVDecoder
from ecgxaiY.utils.loss import CombinedLoss, GaussianVAEReconLoss, KLDivergence
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def run_trainer(params):
    pl.seed_everything(1234)

    # don't forget to update this for your own logger
    api_key = open("neptune_token.txt", "r").read()
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project=params['training']['project2_name'],
        tags=params['training']['tags'],
        source_files=['*.py', '*.json', '*.yaml', '../../ecgxai/**/*.py']
    )
    neptune_logger.experiment["model/hyper-parameters"] = params

    # define transforms
    transform = transforms.Compose([ApplyGain(), ToTensor()])

    # define datasets
    traindf = pd.read_csv(params["paths"]["training_labels"])
    traindf = traindf[traindf['stage'].str.contains('STAGE 2')]
    trainset = UniversalECGDataset(
        'umcu',
        params["paths"]["raw_data"],
        traindf,
        transform=transform
    )
    train_loader = DataLoader(
        trainset,
        batch_size=params['training']['batch_size'],
        num_workers=8,
        shuffle=True
    )

    valdf = pd.read_csv(params["paths"]["validation_labels"])
    valdf = valdf[valdf['stage'].str.contains('STAGE 2')]
    valset = UniversalECGDataset(
        'umcu',
        params["paths"]["raw_data"],
        valdf,
        transform=transform
    )
    val_loader = DataLoader(
        valset,
        batch_size=params['training']['batch_size'],
        num_workers=8
    )

    loss = CombinedLoss(
        [
            GaussianVAEReconLoss(reduction='mean'),
            KLDivergence(reduction='mean', std_is_log=False)
        ],
        ['+'],
        [1, 32]
    )

    metrics = MetricCollection({
        'MSE': TMW(MeanSquaredError(), ['x', 'reconstruction']),
    })

    if params['training']['pretrain']:
        model = GaussianVAE.load_from_checkpoint(
            checkpoint_path=params['paths']['pretrain_checkpoint'],
            loss=loss,
            lr=params['training']['learning_rate'],
            train_metrics=metrics,
            val_metrics=metrics,
            std_is_log=False
        )
    else:
        encoder = CausalCNNVEncoder(**params['encoder'])
        decoder = CausalCNNVDecoder(**params['decoder'])

        model = GaussianVAE(
            encoder_class=encoder,
            decoder_class=decoder,
            loss=loss,
            lr=params['training']['learning_rate'],
            train_metrics=metrics,
            val_metrics=metrics,
            std_is_log=False
        )

    trainer = pl.Trainer(
        max_epochs=params['training']['epochs'],
        logger=neptune_logger,
        log_every_n_steps=5,
        gradient_clip_val=1,
        gpus=1,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_last=True,
                dirpath=os.path.join(params['paths']['checkpoints'], neptune_logger.version),
                filename='epoch={epoch}-step={step}-loss={val_loss:.2f}'
            ),
        ]
    )
    trainer.fit(model, train_loader, val_loader)


with open('/home/ubuntu/djboo/FactorECG/train_factorecg_vae_TMT.yaml', "r") as stream:
    params = yaml.safe_load(stream)

print(params)
output = run_trainer(params)
