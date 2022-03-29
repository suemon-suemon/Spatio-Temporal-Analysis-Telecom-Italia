"""
The main code is based on https://github.com/optuna/optuna-examples/blob/63fe36db4701d5b230ade04eb2283371fb2265bf/pytorch/pytorch_simple.py
"""

from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import wandb
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from models import LSTMRegressor
from datasets import MilanSW

MAX_EPOCHS = 200

def create_LSTM_dm(trial):
    emb_size = trial.suggest_int("emb_size", 16, 128, log=True)
    hidden_size = trial.suggest_int("hidden_size", 16, 128, log=True)
    seq_len = trial.suggest_int("seq_len", 12, 144, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    is_input_emb = trial.suggest_categorical("is_input_emb", [True, False])
    trial.set_user_attr("batch_size", 1024)
    print(emb_size, hidden_size, seq_len, num_layers, dropout, learning_rate, is_input_emb)
    dm = MilanSW(batch_size=trial.user_attrs['batch_size'], in_len=seq_len, out_len=1)
    model = LSTMRegressor(n_features=121, 
                         emb_size=emb_size, 
                         hidden_size=hidden_size, 
                         seq_len=seq_len, 
                         criterion=torch.nn.MSELoss(), 
                         num_layers=num_layers, 
                         dropout=dropout, 
                         learning_rate=learning_rate,
                         is_input_embedding=is_input_emb)
    return model, dm


def objective(trial):
    model, dm = create_LSTM_dm(trial)

    logger = WandbLogger(project="spatio-temporal prediction")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger.experiment.config["exp_tag"] = "optuna_search"

    trainer = pl.Trainer(
        logger=logger,
        limit_val_batches=0.1,
        max_epochs=MAX_EPOCHS,
        gpus=1,
        callbacks=[lr_monitor, 
                   EarlyStopping(monitor='val_loss', patience=20), 
                   PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )

    trainer.fit(model, dm)
    val_obj = trainer.callback_metrics["val_loss"].item()
    wandb.finish()
    return val_obj

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    wandb.login()
    
    study = optuna.create_study(
        direction="minimize",
        study_name="milan-LSTM",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_search.png")
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("optuna_importance.png")