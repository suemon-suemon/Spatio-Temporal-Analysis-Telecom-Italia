from asyncio.log import logger
from fix_path import fix_python_path_if_working_locally
fix_python_path_if_working_locally()
import wandb
from datasets import MilanFG
from models import Mvstgn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from torch import nn

if __name__ == "__main__":
    seed_everything(42)
    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530

    p = dict(
        # dataset & model
        aggr_time = '10min',
        time_range = 'all',
        normalize = True,
        batch_size = 32,
        learning_rate = 1e-3,
        close_len = 6,
        period_len = 0,
        trend_len = 0,
        pred_len = 3,
        service_dim = 1,
        # grid_range = (41, 60 ,41, 60),

        # trainer
        max_epochs = 300,
        criterion = nn.L1Loss,

        # logger
        use_wandb = True,  # 设置为False即可禁用wandb
        log_project_name="MilanPredict",
        log_run_name='MVSTGN_6_3',
        log_run_tag=["MVSTGN_6_3"],
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
        service_dim = p['service_dim'],
        # grid_range=p['grid_range'],
    )

    model = Mvstgn(
        input_shape=(p['batch_size'], p['close_len'], dm.service_dim, dm.n_rows, dm.n_cols),
        learning_rate=p['learning_rate'],
        criterion=p['criterion'],
        pred_len = p['pred_len'],
    )
    # model = Mvstgn.load_from_checkpoint("milanST/3eyl8rqy/checkpoints/epoch=203-step=14075.ckpt")

    # ✅ 根据use_wandb变量设置logger
    wandb_logger = None
    if p['use_wandb']:
        wandb.init(
            project=p["log_project_name"],
            name=p["log_run_name"],
            tags=p["log_run_tag"],
            id=f"{p['log_run_name']}_{timestamp}",
            resume="allow",  # 允许恢复，指定 `run_id`
            # id="s827fy4m", resume="must", # 恢复已有的
        )
        wandb.config.update(p, allow_val_change=True)
        # 创建 `WandbLogger`，复用 `wandb` 运行实例
        wandb_logger = WandbLogger(experiment=wandb.run)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs = p['max_epochs'],
        logger = wandb_logger,
        devices = 1,
        enable_model_summary=True,
        enable_checkpointing=True,
        callbacks = [lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )
    # trainer.logger.experiment.save('models/Mvstgn.py')

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm)
    