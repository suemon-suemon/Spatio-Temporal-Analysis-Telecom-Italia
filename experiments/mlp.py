from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanFG
from models import MLP
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
        tele_col = 'internet',
        batch_size = 64,
        learning_rate = 1e-4,
        normalize = False,
        
        # model trainer
        max_epochs = 1000,
        criterion = nn.L1Loss,
        close_len = 12,
        period_len = 0,
        pred_len = 1,
        mlp_dim = 256,
    )

    model = MLP(
        mlp_dim=p['mlp_dim'],
        pred_len=p['pred_len'],
        close_len=p['close_len'],
        period_len=p['period_len'],
        learning_rate=p['learning_rate'],
    )

    dm = MilanFG(
        tele_column=p['tele_col'],
        batch_size=p['batch_size'], 
        close_len=p['close_len'], 
        period_len=p['period_len'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
        pred_len=p['pred_len'],
    )

    wandb_logger = WandbLogger(name=f"MLP_{'hr' if p['aggr_time']=='hour' else 'min'}_in{p['close_len']}+{p['period_len']}_pred{p['pred_len']}_{p['tele_col']}", 
                               project="milanST")
    wandb_logger.experiment.config["exp_tag"] = "MLP"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )
    trainer.logger.experiment.save('models/MLP.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
