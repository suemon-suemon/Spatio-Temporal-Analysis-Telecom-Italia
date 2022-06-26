from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanSW
from models import STTran
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

if __name__ == "__main__":
    seed_everything(42)

    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = 'hour',
        batch_size = 512,
        learning_rate = 1e-3,
        normalize = True,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss,

        pred_len = 1,
        close_len = 3,
        period_len = 3,
        k_grids = 20,
    )

    model = STTran(
        close_len = p['close_len'],
        period_len = p['period_len'], 
        pred_len = p['pred_len'],
        k_grids = p['k_grids'],
        learning_rate = p['learning_rate'],
        criterion = p['criterion']
    )

    dm = MilanSW(
        format = 'sttran',
        batch_size=p['batch_size'],
        close_len=p['close_len'], 
        period_len=p['period_len'], 
        out_len = p['pred_len'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
    )
    
    wandb_logger = WandbLogger(name='stTran_in3_pred1_min', project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "StTran"
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
