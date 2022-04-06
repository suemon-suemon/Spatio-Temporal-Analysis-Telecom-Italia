from pytorch_lightning import LightningModule
from torch import from_numpy, cat
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError,
                          MeanSquaredError,
                          SymmetricMeanAbsolutePercentageError)
import matplotlib.pyplot as plt
from wandb import wandb
from utils.nrmse import nrmse

class STBase(LightningModule):
    def __init__(self,
                 learning_rate: float = 1e-5,
                 criterion = L1Loss(),
                 reduceLRPatience: int = 5,):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.reduceLRPatience = reduceLRPatience
        self.save_hyperparameters()

        self.valid_MAE = MeanAbsoluteError()
        self.valid_MAPE = MeanAbsolutePercentageError()
        self.valid_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.valid_RMSE = MeanSquaredError(squared=False)
        self.test_MAE = MeanAbsoluteError()
        self.test_MAPE = MeanAbsolutePercentageError()
        self.test_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.test_RMSE = MeanSquaredError(squared=False)
        self.test_NRMSE = MeanSquaredError(squared=False)

    def forward(self, x):
        raise NotImplementedError
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.reduceLRPatience)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)
        
        self.valid_MAE(y_hat, y)
        self.log('val_MAE', self.valid_MAE, on_epoch=True)
        self.valid_MAPE(y_hat, y)
        self.log('val_MAPE', self.valid_MAPE, on_epoch=True)
        self.valid_RMSE(y_hat, y)
        self.log('val_RMSE', self.valid_RMSE, on_epoch=True)
        self.valid_SMAPE(y_hat, y)
        self.log('val_SMAPE', self.valid_SMAPE, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)

        self.log('test_loss', loss)
        self.test_MAE(y_hat, y)
        self.log('test_MAE', self.test_MAE, on_epoch=True)
        self.test_MAPE(y_hat, y)
        self.log('test_MAPE', self.test_MAPE, on_epoch=True)
        self.test_RMSE(y_hat, y)
        self.log('test_RMSE', self.test_RMSE, on_epoch=True)
        self.test_SMAPE(y_hat, y)
        self.log('test_SMAPE', self.test_SMAPE, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        y_hat = self._inverse_transform(y_hat) if self.trainer.datamodule.normalize else y_hat
        return y_hat

    def on_predict_epoch_end(self, results):
        preds = cat(results[0]).reshape(-1, self.trainer.datamodule.n_grids)
        gt = self.trainer.datamodule.milan_test[self.seq_len:].reshape(-1, self.trainer.datamodule.n_grids)
        self.log('test_nrmse', nrmse(preds, gt))
        if self.trainer.datamodule.normalize:
            gt = self.trainer.datamodule.scaler.inverse_transform(gt.reshape(-1, 1)).reshape(gt.shape)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                # compare preds with ground truth, draw plots
                base = 300
                axes[i, j].plot(gt[:, base+i*3+j], label="ground truth")
                axes[i, j].plot(preds[:, base+i*3+j], label="prediction")
                axes[i, j].legend()
        self.logger.log_image('predictions', [wandb.Image(fig)])
        plt.savefig("preds_mvstgn_{}.png".format(self.logger.version))

    def _inverse_transform(self, y):
        yn = y.detach().cpu().numpy() # detach from computation graph
        scaler = self.trainer.datamodule.scaler
        yn = scaler.inverse_transform(yn.reshape(-1, 1)).reshape(yn.shape)
        return from_numpy(yn).cuda()
