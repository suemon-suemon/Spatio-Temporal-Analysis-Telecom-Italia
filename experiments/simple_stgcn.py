# from networkx.readwrite.json_graph.adjacency import adjacency_graph
import networkx as nx
from fix_path import fix_python_path_if_working_locally
fix_python_path_if_working_locally()
import torch
import numpy as np
from datasets import MilanFG
from models.SimpleSTGCN import SimpleSTGCN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from datetime import datetime


if __name__ == "__main__":

    seed_everything(42)

    p = dict(
        # dataset
        aggr_time = '10min',
        time_range = 'all',
        normalize = True,
        batch_size = 32,
        learning_rate = 1e-3,
        feature_len = 64,
        cheb_k = 2,

        max_epochs = 1000,
        criterion = nn.L1Loss,
        close_len = 6,
        pred_len = 3,
        period_len = 0,
        trend_len = 0,
    )

    dm = MilanFG(
        format='stgcn',
        batch_size=p['batch_size'],
        close_len=p['close_len'],
        period_len=p['period_len'],
        trend_len=p['trend_len'],
        pred_len=p['pred_len'],
        normalize=p['normalize'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
    )

    model = SimpleSTGCN(
        close_len = p['close_len'],
        pred_len = p['pred_len'],
        feature_len = p['feature_len'],
        cheb_k = p['cheb_k'],
    )

    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530
    wandb_logger = WandbLogger(project="MilanPredict",
                            name="SimpleSTGCN_GridGs_LPGt_GCN_6_3",
                            id=f"{'SimpleSTGCN_GridGs_LPGt_GCN_6_3'}_{timestamp}", )
    wandb_logger.experiment.config["exp_tag"] = ["lightstgcn"]
    wandb_logger.experiment.config.update(p, allow_val_change=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        logger= wandb_logger,
        devices = 1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=10,
                                             min_delta=1e-6, verbose=True)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
