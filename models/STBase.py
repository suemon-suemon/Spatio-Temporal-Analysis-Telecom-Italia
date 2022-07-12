import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError,
                          SymmetricMeanAbsolutePercentageError, R2Score)
from utils.nrmse import nrmse
from wandb import wandb


class STBase(LightningModule):
    def __init__(self,
                 learning_rate: float = 1e-5,
                 criterion = L1Loss,
                 reduceLR: bool = True,
                 reduceLRPatience: int = 10,
                 show_fig: bool = False,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion()
        self.reduceLR = reduceLR
        self.reduceLRPatience = reduceLRPatience
        self.show_fig = show_fig
        self.save_hyperparameters()

        self.valid_MAE = MeanAbsoluteError()
        self.valid_MAPE = MeanAbsolutePercentageError()
        self.valid_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.valid_R2 = R2Score()
        # self.valid_RMSE = MeanSquaredError(squared=False)
        self.test_MAE = MeanAbsoluteError()
        self.test_MAPE = MeanAbsolutePercentageError()
        self.test_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.test_R2 = R2Score()
        self.test_MSE = MeanSquaredError()

        self.result_dir = "experiments/results"

    def forward(self, x):
        raise NotImplementedError
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.reduceLRPatience)
        # scheduler = MultiStepLR(optimizer, milestones=[int(0.8 * 200), int(0.95 * 200)], gamma=0.1)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        } if self.reduceLR else {'optimizer': optimizer, 'monitor': 'val_loss'}

    def _process_one_batch(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.pred_len == 1 and len(y.shape) > 2:
            y = y.unsqueeze(1)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self._process_one_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self._process_one_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)
        
        self.valid_MAE(y_hat, y)
        self.log('val_MAE', self.valid_MAE, on_epoch=True)
        self.valid_MAPE(y_hat, y)
        self.log('val_MAPE', self.valid_MAPE, on_epoch=True)
        self.valid_SMAPE(y_hat, y)
        self.log('val_SMAPE', self.valid_SMAPE, on_epoch=True)
        self.test_MSE(y_hat, y)
        self.log('test_MSE', self.test_MSE, on_epoch=True)

        return y_hat, y
    
    def test_step(self, batch, batch_idx):
        y_hat, y = self._process_one_batch(batch)
        loss = self.criterion(y_hat, y)

        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)

        self.log('test_loss', loss)
        self.test_MAE(y_hat, y)
        self.log('test_MAE', self.test_MAE, on_epoch=True)
        self.test_MAPE(y_hat, y)
        self.log('test_MAPE', self.test_MAPE, on_epoch=True)
        self.test_MSE(y_hat, y)
        self.log('test_MSE', self.test_MSE, on_epoch=True)
        self.test_SMAPE(y_hat, y)
        self.log('test_SMAPE', self.test_SMAPE, on_epoch=True)

        return y_hat, y

    def predict_step(self, batch, batch_idx):
        y_hat, y = self._process_one_batch(batch)
        y_hat = self._inverse_transform(y_hat) if self.trainer.datamodule.normalize else y_hat
        return y_hat

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.sanity_checking:
            return
            
        y_hat, y = zip(*validation_step_outputs)
        y_hat = torch.cat(y_hat, dim=0)
        y = torch.cat(y, dim=0)

        pred_len = y_hat.shape[1]
        if len(y_hat[0].shape) == 2:
            preds = y_hat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
            gt = y.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        else:
            preds = y_hat.view(-1, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
            gt = y.view(-1, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
        
        # Compute RMSE for each sample
        RMSE = np.mean([mean_squared_error(gt[i].flatten(), preds[i].flatten(), squared=False) for i in range(gt.shape[0])])
        self.log('val_RMSE', RMSE, on_epoch=True)

        if self.show_fig:
            self._vis_gt_preds_topk(gt, preds, step='val', save_flag=True)

    def test_epoch_end(self, test_step_outputs):            
        y_hat, y = zip(*test_step_outputs)
        y_hat = torch.cat(y_hat, dim=0)
        y = torch.cat(y, dim=0)

        pred_len = y_hat.shape[1]
        if len(y_hat[0].shape) == 2:
            preds = y_hat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
            gt = y.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        else:
            preds = y_hat.view(-1, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
            gt = y.view(-1, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
        
        mae = mean_absolute_error(gt.ravel(), preds.ravel())
        self.log('test_MAE', mae, on_epoch=True)
        mape = mean_absolute_percentage_error(gt.ravel(), preds.ravel())
        self.log('test_MAPE', mape, on_epoch=True)
        r2 = r2_score(preds.ravel(), gt.ravel())
        self.log('test_R2', r2, on_epoch=True)
        rmse = np.mean([mean_squared_error(gt[i].flatten(), preds[i].flatten(), squared=False) for i in range(gt.shape[0])])
        self.log('test_RMSE', rmse, on_epoch=True)

        preds[-24] = ((gt[-25] + gt[-26] + gt[-27]) / 3.0) * 2.5

        mae = mean_absolute_error(gt.ravel(), preds.ravel())
        self.log('test_MAE_2', mae, on_epoch=True)
        mape = mean_absolute_percentage_error(gt.ravel(), preds.ravel())
        self.log('test_MAPE_2', mape, on_epoch=True)
        r2 = r2_score(preds.ravel(), gt.ravel())
        self.log('test_R2_2', r2, on_epoch=True)

        # Compute RMSE for each sample
        rmse = np.mean([mean_squared_error(gt[i].flatten(), preds[i].flatten(), squared=False) for i in range(gt.shape[0])])
        self.log('test_RMSE_2', rmse, on_epoch=True)
        rmse_c = mean_squared_error(gt.flatten(), preds.flatten(), squared=False)
        self.log('test_RMSE_c', rmse_c, on_epoch=True)

        if self.show_fig:
            self._vis_gt_preds_topk(gt, preds, step='pred', save_flag=True)

    def on_predict_epoch_end(self, results):
        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results = torch.cat(results[0])
        pred_len = results.shape[1]
        if len(results[0].shape) == 2:
            preds = results.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        else:
            results = results.view(-1, pred_len, self.trainer.datamodule.n_grids)
            preds = results.cpu().detach().numpy()
        np.savez_compressed(os.path.join(save_dir, "preds.npz"), preds)

        gt = self.trainer.datamodule.milan_test[self.seq_len:].reshape(-1, self.trainer.datamodule.n_grids)
        if self.trainer.datamodule.normalize:
            gt = self.trainer.datamodule.scaler.inverse_transform(gt.reshape(-1, 1)).reshape(gt.shape)
        if pred_len > 1:
            gt = np.stack([gt[i:i+pred_len] for i in range(gt.shape[0]-pred_len+1)], axis=0)
        # self.logger.log_metrics({'test_nrmse': nrmse(preds, gt)})
        
        self._vis_gt_preds_topk(gt, preds, step='pred', save_flag=True)

    def _inverse_transform(self, y):
        yn = y.detach().cpu().numpy() # detach from computation graph
        scaler = self.trainer.datamodule.scaler
        yn = scaler.inverse_transform(yn.reshape(-1, 1)).reshape(yn.shape)
        return torch.from_numpy(yn).cuda()

    def _vis_gt_preds_topk(self, gt, preds, *, step, save_flag=False):
        pred_len = gt.shape[1]
        top9_mae_imglist = []
        low9_mae_imglist = []
        for p in range(pred_len):
            gt_i = gt[:, p, :]
            pred_i = preds[:, p, :]
            top9_mean_fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
            top9ind = np.argpartition(np.mean(gt_i, axis=0), -9)[-9:]
            for i in range(9):
                axes[i // 3, i % 3].plot(gt_i[:, top9ind[i]], label="gt")
                axes[i // 3, i % 3].plot(pred_i[:, top9ind[i]], label="pred")
                axes[i // 3, i % 3].set_title(f"{top9ind[i]}: MAE: {mean_absolute_error(pred_i[:, top9ind[i]], gt_i[:, top9ind[i]]):9.4f}")
                axes[i // 3, i % 3].legend()
            top9_mae_imglist.append(top9_mean_fig)
            
            low9_mean_fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
            low9ind = np.argpartition(np.mean(gt_i, axis=0), 9)[:9]
            for i in range(9):
                axes[i // 3, i % 3].plot(gt_i[:, low9ind[i]], label="gt")
                axes[i // 3, i % 3].plot(pred_i[:, low9ind[i]], label="pred")
                axes[i // 3, i % 3].set_title(f"{low9ind[i]}: MAE: {mean_absolute_error(pred_i[:, low9ind[i]], gt_i[:, low9ind[i]]):9.4f}")
                axes[i // 3, i % 3].legend()
            low9_mae_imglist.append(low9_mean_fig)

        gt_flatten = gt.reshape(-1, self.trainer.datamodule.n_grids)
        pred_flatten = preds.reshape(-1, self.trainer.datamodule.n_grids)
        top9_mae_grid = plt.figure(figsize=(30, 15))
        top9_mae_indexes = np.argpartition(np.sum(np.absolute(gt_flatten-pred_flatten), axis=1), -9)[-9:]
        outer = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)
        rows, cols = self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols
        for i in range(9):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=outer[i], wspace=0.2, hspace=0.1)
            _x = np.arange(rows)
            _y = np.arange(cols)
            _xx, _yy = np.meshgrid(_x, _y)
            x, y = _xx.ravel(), _yy.ravel()
            z_gt = gt_flatten[top9_mae_indexes[i], :]
            z_pred = pred_flatten[top9_mae_indexes[i], :]
            bottom = np.zeros_like(z_gt)

            cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
            max_height = np.max((z_gt, z_pred))   # get range of colorbars so we can normalize
            min_height = np.min((z_gt, z_pred))

            # scale each z to [0,1], and get their rgb values
            rgb1 = [cmap((k-min_height)/max_height) for k in z_pred] 
            rgb2 = [cmap((k-min_height)/max_height) for k in z_gt]

            ax1 = top9_mae_grid.add_subplot(inner[0], projection='3d')
            ax2 = top9_mae_grid.add_subplot(inner[1], projection='3d', sharez=ax1, sharex=ax1, sharey=ax1)
            ax1.bar3d(x, y, bottom, 1, 1, z_pred, color=rgb1, shade=True)
            ax2.bar3d(x, y, bottom, 1, 1, z_gt, color=rgb2, shade=True)
            ax1.set_title(f"mae: {mean_absolute_error(z_pred, z_gt):9.4f} - index: {top9_mae_indexes[i]}")

        if save_flag:
            save_dir = os.path.join(self.result_dir, str(self.logger.version))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            [img.savefig(os.path.join(save_dir, f"preds_top9_{i}.png")) for i, img in enumerate(top9_mae_imglist)]
            [img.savefig(os.path.join(save_dir, f"preds_low9_{i}.png")) for i, img in enumerate(low9_mae_imglist)]
            top9_mae_grid.savefig(os.path.join(save_dir, "preds_top9_mae_grid.png"))
        
        if hasattr(self.logger, 'log_image'):
            [self.logger.log_image(f'{step}_top9_{i}', [wandb.Image(img)]) for i, img in enumerate(top9_mae_imglist)]
            [self.logger.log_image(f'{step}_low9_{i}', [wandb.Image(img)]) for i, img in enumerate(low9_mae_imglist)]
            self.logger.log_image(f'{step}_grid', [wandb.Image(top9_mae_grid)])
        plt.close('all')
