from fix_path import fix_python_path_if_working_locally
from datetime import datetime
fix_python_path_if_working_locally()
import torch
import wandb
from datasets import MilanFG, MilanSW
from models.SCOPE import SCOPE
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

if __name__ == "__main__":

    seed_everything(42)
    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530

    p = dict(
        log_run_name='SCOPE_sms_downsam16_512_64',
        log_project_name="MilanPredict",
        log_run_tag=["one service", "complex", "down_sampling=1", "node independent"],
        time_now=timestamp,
        show_fig=True,
        show_intermediate_results=True,

        # dataset
        aggr_time = '10min',
        time_range = 'all',
        normalize = True,
        tele_column = 'sms',
        batch_size = 256,
        learning_rate = 1e-2,
        period_len = 0,
        trend_len = 0,

        # 模型
        max_epochs = 1000,
        criterion = nn.L1Loss(),
        close_len = 512,
        pred_len = 64,
        downsampling_rate = 16,
        time_basis_number = 16,
    )

    wandb.init(
        project=p["log_project_name"],
        name=p["log_run_name"],
        tags=p["log_run_tag"],
        id=f"{p['log_run_name']}_{timestamp}",
        resume="allow",  # 允许恢复，指定 `run_id`
        # id="s827fy4m", resume="must", # 恢复已有的
    )
    wandb.config.update(p, allow_val_change = True)

    # 创建 `WandbLogger`，复用 `wandb` 运行实例
    wandb_logger = WandbLogger(experiment=wandb.run)

    dm = MilanFG(
        format='scope',
        batch_size = p['batch_size'],
        close_len = p['close_len'],
        period_len = p['period_len'],
        trend_len = p['trend_len'],
        pred_len = p['pred_len'],
        normalize = p['normalize'],
        aggr_time = p['aggr_time'],
        time_range = p['time_range'],
    )

    model = SCOPE(
            time_basis_number = p['time_basis_number'],
            downsampling_rate = p['downsampling_rate'],
            close_len = p['close_len'],
            pred_len = p['pred_len'],
            show_fig = p['show_fig'],
            show_intermediate_results = p['show_intermediate_results'],
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        log_every_n_steps = 1,
        check_val_every_n_epoch = 1,
        max_epochs = p['max_epochs'],
        logger = wandb_logger,
        devices = 1,
        enable_progress_bar = True,
        enable_model_summary = True,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=50, verbose=True)]
    )

    # model = SCOPE.load_from_checkpoint("experiments/lightning_logs/SCOPE_sms_downsam4_03221638/checkpoints/epoch=999-step=26000.ckpt")
    torch.autograd.set_detect_anomaly(True)  # ✅ 全局启用梯度异常检测
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)



