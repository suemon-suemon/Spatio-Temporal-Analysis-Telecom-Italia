from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import time

from datasets.milan import Milan
from models.lstm import LSTMRegressor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch import nn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

seed_everything(42)
wandb_logger = WandbLogger(project="spatio-temporal prediction")

p = dict(
    seq_len = 12,
    criterion = nn.MSELoss(),
    max_epochs = 10,
    n_features = 121,
    hidden_size = 64,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)

trainer = Trainer(
    max_epochs=p['max_epochs'],
    logger=wandb_logger,
    gpus=1,
)

model = LSTMRegressor(
    n_features = p['n_features'],
    hidden_size = p['hidden_size'],
    seq_len = p['seq_len'],
    criterion = p['criterion'],
    num_layers = p['num_layers'],
    dropout = p['dropout'],
    learning_rate = p['learning_rate']
)

dm = Milan()

trainer.fit(model, dm)
trainer.test(model, datamodule=dm)
