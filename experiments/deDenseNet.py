from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanFG
from models import DeDenseNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":

    seed_everything(42)
    for tele_col in ['sms', 'call', 'internet']:
        p = dict(
            # dataset
            time_range = 'all',
            aggr_time = 'hour',
            tele_col = tele_col,
            grid_range = (41, 60, 41, 60),

            normalize = True,
            batch_size = 32,
            learning_rate = 1e-3,
    
            max_epochs = 500,
            criterion = nn.L1Loss,
            close_len = 96,
            period_len = 0,
            trend_len = 0,
            pred_len = 1,

            layers_s = 4,
            growth_rate_s = 32,
            num_init_features_s = 32,
            bn_size_s = 4,

            layers_t = 4,
            growth_rate_t = 32,
            num_init_features_t = 32,
            bn_size_t = 4,

            drop_rate = 0.3,
            kernel_size = 25,
        )

        dm = MilanFG(
            grid_range=p['grid_range'],
            batch_size=p['batch_size'],
            close_len=p['close_len'], 
            period_len=p['period_len'], 
            trend_len=p['trend_len'],
            pred_len=p['pred_len'],
            normalize=p['normalize'],
            aggr_time=p['aggr_time'],
            time_range=p['time_range'],
            tele_column=p['tele_col'],
        )
        model = DeDenseNet(
            learning_rate = p['learning_rate'],
            channels = p['close_len'],

            layers_s = p['layers_s'],
            growth_rate_s = p['growth_rate_s'],
            num_init_features_s = p['num_init_features_s'],
            bn_size_s = p['bn_size_s'],

            layers_t = p['layers_t'],
            growth_rate_t = p['growth_rate_t'],
            num_init_features_t = p['num_init_features_t'],
            bn_size_t = p['bn_size_t'],
            
            drop_rate = p['drop_rate'],
            kernel_size = p['kernel_size'],
        )

        # wandb_logger = WandbLogger(project="milanST",
        #     name=f"STDense_in{p['close_len']}_out{p['pred_len']}_{'hr' if p['aggr_time'] == 'hour' else 'min'}_{p['time_range']}")
        # wandb_logger.experiment.config["exp_tag"] = "stDenseNet"
        # wandb_logger.experiment.config.update(p)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            max_epochs=p['max_epochs'],
            # logger=wandb_logger,
            gpus=1,
            callbacks=[lr_monitor, 
                        EarlyStopping(monitor='val_loss', patience=15)
                        ]
        )    
        # trainer.logger.experiment.save('models/STDenseNet.py')

        trainer.fit(model, dm)
        trainer.test(model, datamodule=dm)
        # trainer.predict(model, datamodule=dm)
