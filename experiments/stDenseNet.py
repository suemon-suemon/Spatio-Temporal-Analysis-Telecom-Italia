from encodings import normalize_encoding
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import torch

import matplotlib.pyplot as plt
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
    # wandb_logger = WandbLogger(project="spatio-temporal prediction")
    # wandb_logger.experiment.config["exp_tag"] = "stDenseNet"

    p = dict(
        # dataset
        aggr_time = 'hour',
        time_range = 'all',
        normalize = True,
        max_norm = 10,
        batch_size = 64,
        learning_rate = 1e-4,

        max_epochs = 500,
        criterion = nn.MSELoss(),
        close_len = 3,
        period_len = 3,
        trend_len = 0,
    )

    dm = MilanFG(
        batch_size=p['batch_size'],
        close_len=p['close_len'], 
        period_len=p['period_len'], 
        trend_len=p['trend_len'],
        normalize=p['normalize'],
        max_norm=p['max_norm'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
    )
    model = STDenseNet(
        learning_rate = p['learning_rate'],
        channels = [p['close_len'], p['period_len'], p['trend_len']]
    )

    # wandb_logger.experiment.config.update(p)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        # logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=30, verbose=True)]
    )    

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    # preds = trainer.predict(model, datamodule=dm, ckpt_path="spatio-temporal prediction/22rb7uun/checkpoints/epoch=59-step=1079.ckpt")
    # preds = torch.cat(preds).reshape(-1, 900)
    # gt = dm.milan_test[p['close_len']:].reshape(-1, 900)
    # gt = dm.scaler.inverse_transform(gt.reshape(-1, 1)).reshape(gt.shape)
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    # for i in range(3):
    #     for j in range(3):
    #         # compare preds with ground truth, draw plots
    #         base = 300
    #         axes[i, j].plot(gt[:, base+i*3+j], label="ground truth")
    #         axes[i, j].plot(preds[:, base+i*3+j], label="prediction")
    #         axes[i, j].legend()
    # plt.savefig("preds.png")