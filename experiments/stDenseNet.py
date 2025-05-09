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

    # pred_len 仅能是 1，改成其他数字就会报错
    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = '10min',

        normalize = True,
        batch_size = 32,
        learning_rate = 1e-3,

        max_epochs = 100,
        criterion = nn.L1Loss,
        close_len = 6,  # 3
        period_len = 0,  # 3
        trend_len = 0,
        pred_len = 3,  # 1

        show_fig = False,
    )

    dm = MilanFG(
        batch_size=p['batch_size'],
        close_len=p['close_len'], 
        period_len=p['period_len'], 
        trend_len=p['trend_len'],
        pred_len=p['pred_len'],
        normalize=p['normalize'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
    )

    model = STDenseNet(
        learning_rate = p['learning_rate'],
        channels = [p['close_len'], p['period_len'], p['trend_len']],
        pred_len=p['pred_len'],
        show_fig = p['show_fig'],
    )

    wandb_logger = WandbLogger(project="MilanPredict",
        name=f"STDense_6_3")
    wandb_logger.experiment.config["exp_tag"] = "stDenseNet"
    wandb_logger.experiment.config.update(p)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=p['max_epochs'],
        enable_model_summary=True,
        enable_progress_bar=True,
        logger=wandb_logger,
        devices=1,
        callbacks=[lr_monitor, 
                    EarlyStopping(monitor='val_loss', patience=10, verbose=True)
                    ]
    )    
    # trainer.logger.experiment.save('models/STDenseNet.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
