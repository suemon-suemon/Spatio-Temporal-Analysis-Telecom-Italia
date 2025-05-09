from fix_path import fix_python_path_if_working_locally
import wandb
fix_python_path_if_working_locally()
import torch
from torch import nn
from datetime import datetime
from datasets import MilanFG
from models.STSGNN import STSGNN
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
        log_run_name='stsgnn_6_3',
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

        max_epochs = 300,
        criterion = nn.L1Loss(),
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
        format='stsgnn',
        batch_size = p['batch_size'],
        close_len = p['close_len'],
        period_len = p['period_len'],
        trend_len = p['trend_len'],
        pred_len = p['pred_len'],
        normalize = p['normalize'],
        aggr_time = p['aggr_time'],
        time_range = p['time_range'],
        )

    model = STSGNN(
                num_nodes = dm.N_all,  # 节点数
                input_dim = 1, # 输入业务种类数
                horizon = p['close_len'],
                output_window = p['pred_len'],
                output_dim = 1, # 输出业务种类数
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
                   EarlyStopping(monitor='val_loss', patience=10,
                                 min_delta=1e-5, verbose=True)]
         )

    # model = MyWAT.load_from_checkpoint("MilanPredict/s827fy4m/checkpoints/epoch=66-step=13869.ckpt", map_location=torch.device("cpu"))

    torch.autograd.set_detect_anomaly(True)  # ✅ 全局启用梯度异常检测
    trainer.fit(model, dm)
    trainer.test(model, dm)
