from pytorch_lightning import LightningModule
from torch import from_numpy
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError,
                          MeanSquaredError,
                          SymmetricMeanAbsolutePercentageError)


class STBase(LightningModule):
    def __init__(self,
                 normalize: bool = False,
                 learning_rate: float = 1e-5,
                 criterion = L1Loss(),
                 reduceLRPatience: int = 5,):
        super().__init__()
        self.normalize = normalize
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

    def forward(self, x):
        raise NotImplementedError
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
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

        if self.normalize:
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

        if self.normalize:
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
        x, y = batch
        y_hat = self(x)
        y_hat = self._inverse_transform(y_hat) if self.normalize else y_hat
        return y_hat

    def _inverse_transform(self, y):
        yn = y.detach().cpu().numpy() # detach from computation graph
        scaler = self.trainer.datamodule.scaler
        yn = scaler.inverse_transform(yn.reshape(-1, 1)).reshape(yn.shape)
        return from_numpy(yn).cuda()
