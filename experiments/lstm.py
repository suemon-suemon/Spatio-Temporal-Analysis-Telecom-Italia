from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os

from datasets import MilanSW
from models import LSTMRegressor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    seed_everything(42)
    # wandb_logger = WandbLogger(project="spatio-temporal prediction", id='3mktckfm', resume=True)

    p = dict(
        batch_size = 1024,
        learning_rate = 1e-3,
        max_epochs = 2,
        criterion = nn.MSELoss(),

        n_features = 121,        
        seq_len = 121,
        emb_size = 74,
        hidden_size = 35,
        num_layers = 3,
        dropout = 0.2081,
    )

    dm = MilanSW(
        batch_size=p['batch_size'], 
        in_len=p['seq_len'], 
        out_len=1
    )
    model = LSTMRegressor(
        n_features = p['n_features'],
        emb_size=p['emb_size'],
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate']
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        # logger=wandb_logger,
        gpus=1,
        callbacks=[lr_monitor, EarlyStopping(monitor='val_loss', patience=20)]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    # # calculate metrics
    # preds = trainer.predict(model, datamodule=dm)
    # preds = torch.cat(preds).reshape(-1, 900)
    # gt = dm.milan_test[p['seq_len']:].reshape(-1, 900)
    # mae = np.mean([mean_absolute_error(gt[i, :], preds[i, :]) for i in range(gt.shape[0])])
    # mape = np.mean([mean_absolute_percentage_error(gt[i, :], preds[i, :]) for i in range(gt.shape[0])])
    # nrmse = nrmse(gt, preds)

    # print('Method: ', 'LSTM', 'MAE: ', mae, ' MAPE: ', mape, ' NRMSE: ', nrmse)
