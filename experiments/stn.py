from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanSW
from models import STN, STN_ConvLSTM, STN_3dConv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":
    seed_everything(42)

    p = dict(
        # dataset
        time_range = '30days',
        aggr_time = None,
        tele_col = 'internet',
        batch_size = 512,
        learning_rate = 1e-4,
        normalize = False,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss,
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
        tele_column = p['tele_col'],
        batch_size=p['batch_size'], 
        close_len=p['seq_len'], 
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
        flatten=False,
    )

    wandb_logger = WandbLogger(project="milanST",
        name=f"stn_in{p['seq_len']}_out1_{'hr' if p['aggr_time']=='hour' else 'min'}_{p['time_range']}")
    wandb_logger.experiment.config["exp_tag"] = "STN"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=15)]
    )
    trainer.logger.experiment.save('models/STN.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
