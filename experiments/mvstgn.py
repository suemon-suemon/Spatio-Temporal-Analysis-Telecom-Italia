from asyncio.log import logger
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanFG
from models import Mvstgn
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
        batch_size = 16,
        learning_rate = 1e-3,
        # grid_range = (41, 60 ,41, 60),

        max_epochs = 500,
        criterion = nn.L1Loss,
        close_len = 3,
        period_len = 0,
        trend_len = 0,
    )

    dm = MilanFG(
        batch_size=p['batch_size'],
        close_len=p['close_len'], 
        period_len=p['period_len'], 
        trend_len=p['trend_len'],
        normalize=p['normalize'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        # grid_range=p['grid_range'],
    )

    model = Mvstgn(
        input_shape=(p['batch_size'], p['close_len'], 1, dm.n_rows, dm.n_cols),
        learning_rate=p['learning_rate'],
    )
    # model = Mvstgn.load_from_checkpoint("milanST/3eyl8rqy/checkpoints/epoch=203-step=14075.ckpt")


    wandb_logger = WandbLogger(project="milanST")
    wandb_logger.experiment.config["exp_tag"] = "Mvstgn"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=[1],
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )
    trainer.logger.experiment.save('models/Mvstgn.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
    