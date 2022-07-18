from matplotlib import pyplot as plt
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os

from datasets import MilanSW, MilanFG
from models import LSTMRegressor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch


if __name__ == "__main__":
    seed_everything(42)

    p = dict(
        # dataset
        time_range = '30days',
        aggr_time = None,
        tele_col = 'callout',
        batch_size = 512,
        learning_rate = 1e-4,
        
        # model trainer
        max_epochs = 1000,
        criterion = nn.L1Loss,
        window_size = 5, 
        n_features = 25, # window_size^2
        close_len = 6,
        pred_len = 1,
        hidden_size = 256,
        num_layers = 4,
        dropout = 0.2,

        is_input_embedding = False,
        emb_size = 16,
    )

    model = LSTMRegressor(
        n_features = p['n_features'],
        emb_size=p['emb_size'],
        hidden_size = p['hidden_size'],
        seq_len = p['close_len'],
        pred_len = p['pred_len'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate'],
        is_input_embedding = p['is_input_embedding'],
    )
    # model = LSTMRegressor.load_from_checkpoint("milanST/29p5cwoa/checkpoints/epoch=199-step=193599.ckpt")
    
    dm = MilanSW(
        tele_column=p['tele_col'],
        batch_size=p['batch_size'], 
        close_len=p['close_len'], 
        pred_len=p['pred_len'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        window_size=p['window_size'],
    )

    # wandb_logger = WandbLogger(project="milanST",
    #     name=f"LSTM_{p['close_len']}_{p['pred_len']}_{'hr' if p['aggr_time'] == 'hour' else 'min'}_{p['time_range']}_{p['tele_col']}")
    # wandb_logger.experiment.config["exp_tag"] = "LSTM"
    # wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        # logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
