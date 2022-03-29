from encodings import normalize_encoding
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os

from datasets import MilanFG
from models import STDenseNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    seed_everything(42)
    wandb_logger = WandbLogger(project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "stDenseNet"

    p = dict(
        batch_size = 64,
        learning_rate = 1e-4,
        max_epochs = 500,
        criterion = nn.MSELoss(),
        aggr_time = 'hour',

        n_features = 121,        
        close_len = 3,
        period_len = 3,
        trend_len = 0,
        normalize = True,
    )

    dm = MilanFG(
        batch_size=p['batch_size'],
        close_len=p['close_len'], 
        period_len=p['period_len'], 
        trend_len=p['trend_len'],
        normalize=p['normalize'],
        aggr_time=p['aggr_time'],
    )
    model = STDenseNet(
        learning_rate = p['learning_rate'],
        channels = [p['close_len'], p['period_len'], p['trend_len']]
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=100, verbose=True)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)