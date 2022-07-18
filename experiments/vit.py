from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import wandb
from datasets import MilanFG, MilanSW
from models import ViT, ViT_matrix
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

if __name__ == "__main__":
    seed_everything(42)
    # for tele_col in ['smsin', 'smsout', 'callin', 'callout', 'internet']:
        
    p = dict(
        # dataset
        time_range = 'all',
        aggr_time = 'hour',
        tele_col = 'internet',
        compare_mvstgn = True,
        grid_range = (41, 60, 41, 60),
        load_meta = True,

        # data_format = '3comp',
        batch_size = 32,
        learning_rate = 1e-4,
        normalize = False,

        close_len = 12,
        period_len = 0,
        pred_len = 1,
        
        # model trainer
        max_epochs = 500,
        criterion = nn.L1Loss,
        # spatial_window = 11,
        patch_size = (2, 2),
        stride_size = (2, 2),
        padding_size = (0, 0),
        vit_dim = 32,
        vit_heads = 8,
        vit_depth = 3,
        mlp_dim = 512,
        channels_group = 1,
        inner_channels = 16,
        d_decoder = 256,
        conv_channels = 512,
        dccnn_layers = 6,
        dccnn_growth_rate = 128,
        dccnn_init_channels = 128,
        dropout_rate = 0.3,
    )

    model = ViT_matrix(
        # reduceLR = False,
        reduceLRPatience = 10,
        image_size=(p['grid_range'][1]-p['grid_range'][0]+1, p['grid_range'][3]-p['grid_range'][2]+1),
        criterion=p['criterion'],
        learning_rate=p['learning_rate'],
        patch_size=p['patch_size'],
        stride_size=p['stride_size'],
        padding_size=p['padding_size'],
        dim=p['vit_dim'],
        depth=p['vit_depth'],
        heads=p['vit_heads'],
        mlp_dim=p['mlp_dim'],
        pred_len=p['pred_len'],
        close_len=p['close_len'],
        period_len=p['period_len'],
        channels_group=p['channels_group'],
        conv_channels=p['conv_channels'],
        dccnn_layers=p['dccnn_layers'],
        dccnn_growth_rate=p['dccnn_growth_rate'],
        dccnn_init_channels=p['dccnn_init_channels'],
        inner_channels=p['inner_channels'],
        d_decoder=p['d_decoder'],
        dropout=p['dropout_rate'],
    )

    dm = MilanFG(
        compare_mvstgn=p['compare_mvstgn'],
        grid_range=p['grid_range'],
        load_meta=p['load_meta'],
        tele_column=p['tele_col'],
        format='timeF',
        batch_size=p['batch_size'], 
        close_len=p['close_len'], 
        period_len=p['period_len'],
        aggr_time=p['aggr_time'],
        time_range=p['time_range'],
        normalize=p['normalize'],
        pred_len=p['pred_len'],
    )

    wandb_logger = WandbLogger(
        name=f"vit_{'hr' if p['aggr_time']=='hour' else 'min'}_{p['tele_col']}_in{p['close_len']}+{p['period_len']}_pred{p['pred_len']}", 
        project="milanST",
        # version='st1nzjid',
        # resume=True,
    )
    wandb_logger.experiment.config["exp_tag"] = "ViT"
    wandb_logger.experiment.config.update(p, allow_val_change=True)
    wandb.save('models/ViT.py')
    wandb.save('models/ViT_matrix.py')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_top_k=1, save_last=True, monitor="val_loss")
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        log_every_n_steps=10,
        logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, checkpoint_callback, EarlyStopping(patience=15, monitor="val_loss")],
    )
    
    # model = ViT_matrix.load_from_checkpoint('milanST/st1nzjid/checkpoints/epoch=84-step=3059.ckpt')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
    # wandb.finish()
