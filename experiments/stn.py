from matplotlib import pyplot as plt
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os

from datasets import MilanSW
from models import STN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    seed_everything(42)

    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = None,
        batch_size = 512,
        learning_rate = 1e-3,
        normalize = False,
        
        # model trainer
        max_epochs = 200,
        criterion = nn.L1Loss(),
        x_dim = 11, 
        y_dim = 11,
        seq_len = 12,
    )

    model = STN(
        x_dim = p['x_dim'],
        y_dim = p['y_dim'],
        seq_len = p['seq_len'],
        learning_rate = p['learning_rate'],
    )

    dm = MilanSW(
        batch_size=p['batch_size'], 
        in_len=p['seq_len'], 
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
        flatten=False,
    )

    wandb_logger = WandbLogger(project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "STN"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
