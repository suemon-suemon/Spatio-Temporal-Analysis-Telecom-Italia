from matplotlib import pyplot as plt
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanSW, MilanFG
from models import TransformerEncoder
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

if __name__ == "__main__":
    IS_SW = True

    seed_everything(42)

    p = dict(
        # dataset
        time_range = '30days',
        aggr_time = None,
        batch_size = 256,
        learning_rate = 1e-4,
        normalize = True,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss,
        x_dim = 11, 
        y_dim = 11,
        seq_len = 48,
    )

    model = TransformerEncoder(
        seq_len = p['seq_len'],
        d_input = p['x_dim'] * p['y_dim'],

        learning_rate = p['learning_rate'],
        criterion = p['criterion']
    )

    dm = MilanSW(
        format = 'normal',
        batch_size=p['batch_size'], 
        in_len=p['seq_len'], 
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
    )

    wandb_logger = WandbLogger(project="milanST")
    wandb_logger.experiment.config["exp_tag"] = "TransformerE_{}".format('SW' if IS_SW else 'FG')
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=5)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
