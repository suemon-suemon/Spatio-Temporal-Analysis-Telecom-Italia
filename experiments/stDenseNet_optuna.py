from encodings import normalize_encoding
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import optuna
import wandb

from datasets import MilanFG
from models import STDenseNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback
from torch import nn

MAX_EPOCHS = 200

def create_STDenseNet_dm(trial):
    # create model and datamodule for trial
    close_len = trial.suggest_int("close_len", 3, 12)
    period_len = trial.suggest_int("period_len", 0, 9)
    trend_len = trial.suggest_int("trend_len", 0, 9)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    trial.set_user_attr("batch_size", 64)
    trial.set_user_attr("normalize", True)
    trial.set_user_attr("aggr_time", None)
    dm = MilanFG(batch_size=trial.user_attrs['batch_size'], 
                 close_len=close_len,
                 period_len=period_len,
                 trend_len=trend_len,
                 normalize=trial.user_attrs['normalize'],
                 time_range='30days',
                 aggr_time=trial.user_attrs['aggr_time'])
    model = STDenseNet(learning_rate=learning_rate,
                       channels=[close_len, period_len, trend_len])
    return model, dm

def objective(trial):
    model, dm = create_STDenseNet_dm(trial)

    logger = WandbLogger(project="milanST")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger.experiment.config["exp_tag"] = "STDenseNet_30days_search"
    logger.experiment.config.update(trial.params, allow_val_change=True)
    # print("params of trail: ", trial.params)

    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=logger,
        max_epochs=MAX_EPOCHS,
        gpus=1,
        callbacks=[lr_monitor, 
                   EarlyStopping(monitor='val_loss', patience=20, verbose=True), 
                   PyTorchLightningPruningCallback(trial, monitor="val_MAE")],
    )

    trainer.fit(model, dm)
    val_obj = trainer.callback_metrics["val_MAE"].item()
    wandb.finish()
    return val_obj


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    wandb.login()
    
    study = optuna.create_study(
        direction="minimize",
        study_name="milan-STDenseNet_hr",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_search_hr.png")
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("optuna_importance_hr.png")
