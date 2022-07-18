from encodings import normalize_encoding
from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import optuna
import wandb

from datasets import MilanFG
from models import ViT_matrix
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback

MAX_EPOCHS = 200

def create_vit_dm(trial):
    # create model and datamodule for trial
    close_len = trial.suggest_categorical("close_len", [12, 24, 36, 48])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    vit_dim = trial.suggest_categorical("vit_dim", [32, 64, 128, 256, 512, 1024])
    vit_heads = trial.suggest_categorical("vit_heads", [4, 8, 16])
    vit_depth = trial.suggest_int("vit_depth", 1, 8)
    mlp_dim = trial.suggest_int("mlp_dim", 32, 1024, log=True)
    channels_group = trial.suggest_int("channels_group", 1, 4)
    conv_channels = trial.suggest_categorical("conv_channels", [24, 48, 96, 144, 192])
    inner_channels = trial.suggest_int("inner_channels", 8, 128, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    trial.set_user_attr("batch_size", 32)
    trial.set_user_attr("normalize", False)
    trial.set_user_attr("aggr_time", None)
    trial.set_user_attr("tele_col", 'internet')
    trial.set_user_attr("time_range", '30days')

    dm = MilanFG(
        tele_column=trial.user_attrs["tele_col"],
        batch_size=trial.user_attrs['batch_size'], 
        format='timeF',
        close_len=close_len,
        period_len=0,
        normalize=trial.user_attrs['normalize'],
        time_range=trial.user_attrs['time_range'],
        aggr_time=trial.user_attrs['aggr_time'])
        
    model = ViT_matrix(
        image_size=(30, 30),
        patch_size=(3, 3),
        stride_size=(3, 3),
        padding_size=(0, 0),
        dim=vit_dim,
        depth=vit_depth,
        heads=vit_heads,
        mlp_dim=mlp_dim,
        pred_len=1,
        close_len=close_len,
        period_len=0,
        learning_rate=learning_rate,
        channels_group=channels_group,
        conv_channels=conv_channels,
        inner_channels=inner_channels,
        dropout=dropout_rate,
    )
    return model, dm

def objective(trial):
    model, dm = create_vit_dm(trial)

    logger = WandbLogger(project="milanST")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger.experiment.config["exp_tag"] = "vit_optuna"
    logger.experiment.config.update(trial.params, allow_val_change=True)
    print("params of trail: ", trial.params)

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
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    wandb.login()
    
    study = optuna.create_study(
        direction="minimize",
        study_name="milan-vit_smsout_hr",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_search_min.png")
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("optuna_importance_min.png")
