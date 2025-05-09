from fix_path import fix_python_path_if_working_locally
fix_python_path_if_working_locally()
import wandb
from datetime import datetime
from models.ARIMAMultiNode import ARIMAMultiNode, LocalARIMAMultiNode
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from datasets.Milan import Milan


if __name__ == "__main__":
    seed_everything(42)
    timestamp = datetime.now().strftime("%m%d%H%M")  # 例如：03181530

    p = dict(
        # dataset & model
        aggr_time = 'hour',
        time_range = 'all',
        close_len = 6,
        pred_len = 3,
        is_period = False, # 模型是否考虑周期性
        grid_range = (41, 43, 41, 43),

        # logger
        use_wandb = False,  # 设置为 False 即可禁用wandb
        log_project_name="MilanPredict",
        log_run_name='ARIMA_Local_6_3',
        log_run_tag=["ARIMA_Local_6_3"],
    )

    # load data
    dataset = Milan(aggr_time=p['aggr_time'],
                    time_range=p['time_range'],
                    grid_range=p['grid_range'],
                    load_meta=False,
                    normalize=False,
                    )
    dataset.prepare_data()
    dataset.setup()
    [train_len, val_len, _] = dataset.get_default_len()
    X = dataset.milan_grid_data.squeeze().reshape(-1, dataset.N_all)
    X = X[:199, :]

    # wandb logger
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

    # arima_baseline = ARIMAMultiNode(
    #     X = X,
    #     input_len = p['close_len'],
    #     pred_len = p['pred_len'],
    #     train_len = train_len,
    #     is_period = p['is_period'],
    #     val_len = val_len,
    #     period_len = dataset.steps_per_day,
    #     logger = wandb_logger
    # )

    arima_baseline = LocalARIMAMultiNode(
        X=X,
        input_len=p['close_len'],
        pred_len=p['pred_len'],
        logger=wandb_logger
    )

    arima_baseline.run()