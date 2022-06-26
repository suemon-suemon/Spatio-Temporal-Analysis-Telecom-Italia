from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanFG
from models import ASTGCN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":

    seed_everything(42)

    p = dict(
        # dataset
        aggr_time = 'hour',
        time_range = 'all',
        normalize = False,
        batch_size = 32,
        learning_rate = 1e-3,

        max_epochs = 1000,
        criterion = nn.L1Loss,
        close_len = 6,
        period_len = 6,
        trend_len = 0,
        pred_len = 6,

        all_backbones = [
            # {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'time_strides': 1}, # w
            {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'time_strides': 1}, # d
            {'K': 3, 'nb_chev_filter': 64, 'nb_time_filter': 64, 'time_strides': 1}, # h
        ],
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
    model = ASTGCN(
        all_backbones = p['all_backbones'],
        adj_mx = dm.adj_mx,
        in_len = p['close_len'],
        pred_len=p['pred_len'],
        learning_rate = p['learning_rate'], 
        criterion = p['criterion'],
    )

    wandb_logger = WandbLogger(name="Stgcn_in6+3_pred6_hr_HI", project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "Stgcn"
    wandb_logger.experiment.config.update(p)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20, verbose=True)]
    )    

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
