from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import optuna
import wandb

from datasets import MilanFG
from models import DeDenseNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback

MAX_EPOCHS = 300
TELE_COL = 'internet'

def create_STDenseNet_dm(trial):
    # create model and datamodule for trial
    close_len = trial.suggest_categorical("close_len", [12, 24, 36, 48, 60, 72, 84, 96])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    layers_s = trial.suggest_int("layers_s", 1, 6)
    growth_rate_s = trial.suggest_int("growth_rate_s", 16, 128, log=True)
    num_init_features_s = trial.suggest_int("num_init_features_s", 16, 128, log=True)
    bn_size_s = trial.suggest_int("bn_size_s", 2, 4)

    layers_t = trial.suggest_int("layers_t", 2, 6)
    growth_rate_t = trial.suggest_int("growth_rate_t", 16, 128, log=True)
    num_init_features_t = trial.suggest_int("num_init_features_t", 16, 128, log=True)
    bn_size_t = trial.suggest_int("bn_size_t", 2, 4)

    drop_rate = trial.suggest_float("drop_rate", 0.0, 0.4)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 9, 13, 17, 21])

    dm = MilanFG(batch_size=32, 
                 close_len=close_len,
                 period_len=0,
                 trend_len=0,
                 normalize=True,
                 time_range='all',
                 tele_column=TELE_COL,
                 aggr_time='hour')
    model = DeDenseNet(learning_rate=learning_rate,
                      channels=close_len,
                      layers_s=layers_s,
                      growth_rate_s=growth_rate_s,
                      num_init_features_s=num_init_features_s,
                      bn_size_s=bn_size_s,
                      layers_t=layers_t,
                      growth_rate_t=growth_rate_t,
                      num_init_features_t=num_init_features_t,
                      bn_size_t=bn_size_t,
                      drop_rate=drop_rate,
                      kernel_size=kernel_size,
                      )
    return model, dm

def objective(trial):
    model, dm = create_STDenseNet_dm(trial)

    logger = WandbLogger(project="milanST")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger.experiment.config["exp_tag"] = "DeDenseNet_search_all_hour_{}".format(TELE_COL)
    logger.experiment.config.update(trial.params, allow_val_change=True)
    # print("params of trail: ", trial.params)

    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=logger,
        max_epochs=MAX_EPOCHS,
        gpus=1,
        callbacks=[lr_monitor, 
                   EarlyStopping(monitor='val_loss', patience=20), 
                   PyTorchLightningPruningCallback(trial, monitor="val_MAE")],
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)
    val_obj = trainer.callback_metrics["test_MAE"].item()
    wandb.finish()
    return val_obj


if __name__ == "__main__":
    seed_everything(42)
    wandb.login()
    
    study = optuna.create_study(
        direction="minimize",
        study_name="DeDenseNet_search_all_hour_{}".format(TELE_COL),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=400)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_search_{}.png".format(TELE_COL))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("optuna_importance_{}.png".format(TELE_COL))
