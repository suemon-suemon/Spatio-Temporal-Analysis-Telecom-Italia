from fix_path import fix_python_path_if_working_locally
import wandb
fix_python_path_if_working_locally()
import torch
from torch import nn
from datetime import datetime
from datasets import MilanFG
from models.StemGNN import StemGNN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":

    seed_everything(42)
    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530

    p = dict(
        show_fig = True,
        show_intermediate_results = True,
        log_run_name='stemgnn_6_3',
        log_project_name="MilanPredict",
        log_run_tag=["grid Gs", "toeplitz Gt"
                     "6->3"],
        time_now=timestamp,

        # dataset
        aggr_time = '10min',
        time_range = 'all',
        normalize = True,
        batch_size = 32,
        learning_rate = 1e-3,

        # model
        multi_layer = 2,
        stack_cnt = 2,
        units = 32, # 隐藏层数量
        dropout_rate = 0.5,

        max_epochs = 100,
        criterion = nn.L1Loss,
        close_len = 6,
        period_len = 0,
        trend_len = 0,
        pred_len = 3,
        )

    # 先用 `wandb.init()` 初始化 WandB
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

    dm = MilanFG(
        format='mywat',
        batch_size = p['batch_size'],
        close_len = p['close_len'],
        period_len = p['period_len'],
        trend_len = p['trend_len'],
        pred_len = p['pred_len'],
        normalize = p['normalize'],
        aggr_time = p['aggr_time'],
        time_range = p['time_range'],
        )

    model = StemGNN(
                node_cnt = dm.N_all,
                units = p['units'],
                multi_layer = p['multi_layer'],
                stack_cnt = p['stack_cnt'],
                dropout_rate = p['dropout_rate'],
                time_step= p['close_len'],
                horizon = p['pred_len'],
                learning_rate = p['learning_rate'],
                show_fig = p['show_fig'],
                criterion=p['criterion'],
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
        callbacks=[lr_monitor,
                   EarlyStopping(monitor='val_loss', patience=20,
                                 min_delta=1e-5, verbose=True)]
         )

    # model = MyWAT.load_from_checkpoint("MilanPredict/s827fy4m/checkpoints/epoch=66-step=13869.ckpt", map_location=torch.device("cpu"))

    torch.autograd.set_detect_anomaly(True)  # ✅ 全局启用梯度异常检测
    trainer.fit(model, dm)
    trainer.test(model, dm)
