from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanSW, MilanFG
from models import Informer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

if __name__ == "__main__":
    IS_SW = False
    seed_everything(42)

    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = None,
        batch_size = 64,
        learning_rate = 1e-3,
        normalize = True,
        
        # model trainer
        max_epochs = 1000,
        criterion = nn.L1Loss,
        x_dim = 11, 
        y_dim = 11,
        seq_len = 48,

        label_len = 12,
        out_len = 1,

        close_len = 16,
        period_len = 6,
        trend_len = 0,
    )
    if IS_SW:
        model = Informer(
            enc_in = p['x_dim'] * p['y_dim'],
            dec_in = 1,
            c_out = 1,
            seq_len = p['seq_len'],
            label_len = p['label_len'],
            out_len = p['out_len'],
            learning_rate = p['learning_rate'],
            criterion = p['criterion']
        )

        dm = MilanSW(
            format = 'informer',
            batch_size=p['batch_size'], 
            in_len=p['seq_len'], 
            label_len = p['label_len'],
            aggr_time=p['aggr_time'],
            time_range=p['time_range'],
            normalize=p['normalize'],
        )
    else:
        model = Informer(
            enc_in = 400,
            dec_in = 400,
            c_out = 400,
            seq_len = p['close_len'],
            label_len = p['label_len'],
            out_len = p['out_len'],
            learning_rate = p['learning_rate'],
            criterion = p['criterion']
        )

        dm = MilanFG(
            format = 'informer',
            batch_size=p['batch_size'],
            close_len=p['close_len'], 
            period_len=p['period_len'], 
            trend_len=p['trend_len'],
            label_len = p['label_len'],
            aggr_time=p['aggr_time'],
            time_range=p['time_range'],
            normalize=p['normalize'],
        )
        
    # model = Informer.load_from_checkpoint("milanST/36jlvz18/checkpoints/epoch=105-step=1788749.ckpt")
    wandb_logger = WandbLogger(project="MilanPredict", name='informer')
    wandb_logger.experiment.config["exp_tag"] = "Informer_{}".format('SW' if IS_SW else 'FG')
    wandb_logger.experiment.config.update(p, allow_val_change=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        devices=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=10)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
