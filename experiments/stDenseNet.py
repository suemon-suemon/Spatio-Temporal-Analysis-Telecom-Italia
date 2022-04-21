from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanFG
from models import STDenseNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":

    seed_everything(42)

    p = dict(
        # dataset
        aggr_time = None,
        time_range = '30days',
        normalize = True,
        max_norm = 1,
        batch_size = 64,
        learning_rate = 1e-3,

        max_epochs = 500,
        criterion = nn.L1Loss,
        close_len = 6,
        period_len = 0,
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
        channels = [p['close_len'], p['period_len'], p['trend_len']],
    )

    wandb_logger = WandbLogger(project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "stDenseNet"
    wandb_logger.experiment.config.update(p)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=50, verbose=True)]
    )    

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
