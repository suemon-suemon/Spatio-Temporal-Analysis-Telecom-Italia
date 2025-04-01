# from networkx.readwrite.json_graph.adjacency import adjacency_graph
import networkx as nx
from fix_path import fix_python_path_if_working_locally
fix_python_path_if_working_locally()
import torch
import numpy as np
from datasets import MilanFG, MilanSW
from models import ASTGCN
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

        max_epochs = 100,
        criterion = nn.L1Loss,
        close_len = 128,
        period_len = 0,
        trend_len = 0,
        pred_len = 32,
    )

    # 使用临时变量存储 all_backbones，避免污染原始字典
    all_backbones = [
        {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'in_len': p['close_len'], 'time_strides': 1},  # recent
        # {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'in_len': p['period_len'], 'time_strides': 1},  # daily
        # {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'in_len': p['trend_len'], 'time_strides': 1}, # weekend
    ]
    # 将 all_backbones 添加到字典中，而不修改其他键值
    p['all_backbones'] = all_backbones

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

    # adj_knn = np.load("/data/scratch/jiayin/AdjKnn2_D15_Milan10Min_Internet.npy")
    # AdjKnn2_D15_Milan10Min_Internet.npy
    # AdjKnn3_D29_Milan10Min_Internet.npy
    # AdjKnn4_D45_Milan10Min_Internet.npy
    # AdjKnn5_D59_Milan10Min_Internet.npy

    model = ASTGCN(
        all_backbones = p['all_backbones'],
        adj_mx = dm.adj_mx, # grid graph
        # adj_mx = nx.adjacency_matrix(nx.grid_2d_graph(n_rows, n_cols))
        # adj_mx = nx.adjacency_matrix(nx.erdos_renyi_graph(400, 3.8/400)),
        # adj_mx = nx.adjacency_matrix(nx.barabasi_albert_graph(400,2)),
        # adj_mx = adj_knn,
        in_len = p['close_len'],
        trend_len = p['trend_len'],
        period_len = p['period_len'],
        pred_len = p['pred_len'],
        learning_rate = p['learning_rate'], 
        criterion = p['criterion'],
    )

    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530
    wandb_logger = WandbLogger(project="MilanPredict",
                            name="STGCN_noTAt_noSAt_1block",
                            id=f"{'STGCN_noTAt_noSAt_1block'}_{timestamp}", )
    wandb_logger.experiment.config["exp_tag"] = ["stgcn", "k_average=3.8",
                                                 "grid graph", "chev GF K=3",
                                                 "noTAt", "noSAt", "1 modules",
                                                 "1 blocks","6/0/0->6"]
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
