import torch
import pytorch_lightning as pl
import re
import argparse
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from types import SimpleNamespace
from utils.registry import get
from utils.auto_import import auto_import_modules_from
from inspect import signature, getmro
import torch.nn as nn

# 自动导入模型/数据集，触发注册器
auto_import_modules_from("models", "models")
auto_import_modules_from("datasets", "datasets")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=1)
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        raw_cfg = yaml.safe_load(f)

    # 先转成 SimpleNamespace
    cfg = SimpleNamespace(**raw_cfg)

    # 再解析 criterion
    cfg = resolve_criterion(cfg)

    # 保证 ckpt_path 存在（默认 None）
    if not hasattr(cfg, "ckpt_path"):
        cfg.ckpt_path = None
    return cfg

def resolve_criterion(cfg):
    if isinstance(cfg.criterion, str):
        try:
            # 支持 "nn.L1Loss" 或 "L1Loss"
            if cfg.criterion.startswith("nn."):
                cfg.criterion = eval(cfg.criterion)  # unsafe in general, but okay if internal
            else:
                cfg.criterion = getattr(nn, cfg.criterion)
        except AttributeError:
            raise ValueError(f"Unknown loss function: {cfg.criterion}")
    return cfg

def filter_args(cls, param_dict):
    """从类的继承链中提取所有 __init__ 的参数"""
    valid_keys = set()
    for base in getmro(cls):  # 获取包括父类的 MRO 顺序
        if '__init__' in base.__dict__:
            sig = signature(base.__init__)
            valid_keys.update(sig.parameters.keys())
    valid_keys.discard('self')
    return {k: v for k, v in param_dict.items() if k in valid_keys}

def sanitize(s):
    """将字符串中的非法字符（用于 WandB run_id/artifact）替换为下划线"""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def init_wandb_logger(cfg, args):
    """
    初始化 WandB 日志器（WandbLogger）

    参数:
        cfg: SimpleNamespace 类型，包含 config.yaml 加载的配置
        args: argparse 命令行参数，包含 log_wandb 等开关

    返回:
        logger (WandbLogger 或 None)
    """
    if not getattr(args, "log_wandb", False):
        return None

    timestamp = datetime.now().strftime("%m%d%H%M")

    exp_name = sanitize(getattr(cfg, "exp_name", "run"))
    run_id = f"{exp_name}_{timestamp}"

    exp_tag = getattr(cfg, "exp_tag", "")
    tags = [t.strip() for t in exp_tag.split(',') if t.strip()]  # 用原始 tag（空格合法），仅用于展示

    project_name = getattr(cfg, "project_name", "DefaultProject")

    logger = WandbLogger(
        project=project_name,
        name=exp_name,
        id=run_id,
        tags=tags
    )

    return logger

if __name__ == "__main__":
    args = parse_args() # 命令行参数
    cfg = load_config(args.config)  # cfg 是 SimpleNamespace
    # 手动覆盖 config 中的值（如果命令行提供了）
    for k, v in vars(args).items():
        if k != 'config' and v is not None:
            setattr(cfg, k, v)

    pl.seed_everything(42)

    # 动态获取模型和数据集类
    DatasetClass = get(cfg.dataset)
    ModelClass = get(cfg.model)

    # 构建 datamodule 和 model，自动过滤多余参数
    # dm = DatasetClass(**filter_args(DatasetClass, vars(cfg)))
    dm = DatasetClass(**vars(cfg)) # 数据集不过滤
    model = ModelClass(**filter_args(ModelClass, vars(cfg))) # 模型过滤

    # wandb 日志（可选）
    logger = init_wandb_logger(cfg, args)

    # 构建 Trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),  # or a list: [0, 1, 2, 3]
        strategy="ddp",  # 分布式数据并行
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        # profiler="simple",  # 打印每部分耗时
    )

    # 训练与测试
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt_path if cfg.ckpt_path else None)
    trainer.test(model, datamodule=dm, ckpt_path=cfg.ckpt_path if cfg.ckpt_path else None)