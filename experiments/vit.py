from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

from datasets import MilanSW, MilanFG
from models import ViT, ViT_pyramid
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
        # data_format = '3comp',
        batch_size = 32,
        learning_rate = 1e-4,
        normalize = False,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss,
        # spatial_window = 11,
        close_len = 6,
        period_len = 0,
        patch_size = (3, 3),
        stride_size = (3, 3),
        vit_dim = 64,
        vit_heads = 32,
        vit_depth = 6,
        mlp_dim = 64,
        pool = 'cls',
    )

    model = ViT_pyramid(
        image_size=(30, 30),
        patch_size=p['patch_size'],
        stride_size=p['stride_size'],
        dim=p['vit_dim'],
        depth=p['vit_depth'],
        heads=p['vit_heads'],
        mlp_dim=p['mlp_dim'],
        pred_len=1,
        close_len=p['close_len'],
        period_len=p['period_len'],
        learning_rate=p['learning_rate'],
        pool=p['pool'],
    )

    # dm = MilanSW(
    #     format=p['data_format'],
    #     batch_size=p['batch_size'], 
    #     close_len=p['close_len'], 
    #     period_len=p['period_len'],
    #     aggr_time=p['aggr_time'],
    #     time_range=p['time_range'],
    #     normalize=p['normalize'],
    #     window_size=p['spatial_window'],
    #     flatten=False,
    # )

    dm = MilanFG(
        batch_size=p['batch_size'], 
        close_len=p['close_len'], 
        period_len=p['period_len'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
    )

    wandb_logger = WandbLogger(name='vitL_MLP', project="spatio-temporal prediction")
    wandb_logger.experiment.config["exp_tag"] = "ViT"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )
    trainer.logger.experiment.save('models/ViT.py')
    trainer.logger.experiment.save('models/ViT_pyramid.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
