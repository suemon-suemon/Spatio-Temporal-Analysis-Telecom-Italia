from matplotlib import pyplot as plt
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os

from datasets import MilanSW
from models import LSTMRegressor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    seed_everything(42)

    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = 'hour',
        batch_size = 512,
        learning_rate = 0.0001633,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss(),
        n_features = 121,        
        seq_len = 12,
        emb_size = 16,
        hidden_size = 80,
        num_layers = 4,
        dropout = 0.223,
        is_input_embedding = True,
    )

    model = LSTMRegressor(
        n_features = p['n_features'],
        emb_size=p['emb_size'],
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate'],
        is_input_embedding = p['is_input_embedding'],
    )
    # model = LSTMRegressor.load_from_checkpoint("spatio-temporal prediction/29p5cwoa/checkpoints/epoch=199-step=193599.ckpt")
    
    dm = MilanSW(
        batch_size=p['batch_size'], 
        in_len=p['seq_len'], 
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
    )

    wandb_logger = WandbLogger(project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "LSTM"
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

    # draw graph
    preds = trainer.predict(model, datamodule=dm)
    preds = torch.cat(preds).reshape(-1, 900)
    gt = dm.milan_test[p['seq_len']:].reshape(-1, 900)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            # compare preds with ground truth, draw plots
            base = 300
            axes[i, j].plot(gt[:, base+i*3+j], label="ground truth")
            axes[i, j].plot(preds[:, base+i*3+j], label="prediction")
            axes[i, j].legend()
    plt.savefig("preds_lstm.png")