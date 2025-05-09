import os
import gc
import re
import wandb
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from utils.funcs import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError,
                          SymmetricMeanAbsolutePercentageError, R2Score)

def clean_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

class STBase(LightningModule):
    def __init__(self,
                 learning_rate: float = 1e-5,
                 criterion = L1Loss,
                 reduceLR: bool = True,
                 reduceLRPatience: int = 10,
                 nb_flows = 1,
                 show_fig: bool = False,
                 show_intermediate_results: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 结果存储路径
        current_directory = os.getcwd() # 当前目录
        parent_directory = os.path.dirname(current_directory) # 上一级目录
        self.result_dir = os.path.join(parent_directory,"experiments/results")
        os.makedirs(self.result_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.criterion = criterion()
        self.validation_outputs = []
        self.test_outputs = []
        self._predict_results = []
        self.reduceLR = reduceLR
        self.reduceLRPatience = reduceLRPatience
        self.nb_flows = nb_flows
        self.show_fig = show_fig
        self.show_intermediate_results = show_intermediate_results
        self.save_hyperparameters()

        self.valid_MAE = MeanAbsoluteError()
        self.valid_MAPE = MeanAbsolutePercentageError()
        self.valid_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.valid_R2 = R2Score()
        self.valid_RMSE = MeanSquaredError(squared=False)
        self.test_MAE = MeanAbsoluteError()
        self.test_MAPE = MeanAbsolutePercentageError()
        self.test_SMAPE = SymmetricMeanAbsolutePercentageError()
        self.test_R2 = R2Score()
        self.test_MSE = MeanSquaredError()

        # 创建 Wandb 表格，存储测试阶段的所有结果
        self.test_results_table = wandb.Table(columns=[
            "Epoch",
            "test_MAE_agg", "test_MAPE_agg", "test_R2", "test_RMSE",
            "test_MAE_2", "test_MAPE_2", "test_R2_2", "test_RMSE_2",
            "test_RMSE_c"
        ])

    def _get_logger_version(self):
        # 为了兼容 self.logger = None 的情况
        return str(self.logger.version) if self.logger and hasattr(self.logger, "version") else "no_logger"

    def forward(self, x):
        raise NotImplementedError
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.reduceLRPatience, min_lr=1e-5)
        # scheduler = MultiStepLR(optimizer, milestones=[int(0.8 * 200), int(0.95 * 200)], gamma=0.1)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        } if self.reduceLR else {'optimizer': optimizer, 'monitor': 'val_loss'}

    def _process_one_batch(self, batch):
        """
        适配不同 `forward()` 返回格式的通用函数
        - 如果 `forward()` 只返回 `y_hat`，则额外损失为 0
        - 如果 `forward()` 返回 `(y_hat, extra_loss)`，则正确解析
        """
        x, y = batch  # 这里的 x 有可能是 list，会在forward中处理

        y_hat = self(x) # 这里的 self(x) 就是 self.forward(x)
        # print(f"y_hat.shape: {y_hat.shape}, y.shape: {y.shape}")

        #  [B, N, F] → [B, 1, N, F] 或 [B, N, F] → [B, 1, F]
        if self.pred_len == 1 and len(y.shape) > 2:
            y = y.unsqueeze(1)

        if isinstance(y_hat, tuple):
            # forward 返回 (y_hat, loss)
            y_hat, extra_loss = y_hat
        else:
            # forward 只返回 y_hat
            y_hat, extra_loss = y_hat, 0.0

        return y_hat, y, extra_loss

    def training_step(self, batch, batch_idx):
        y_hat, y, extra_loss = self._process_one_batch(batch)
        base_loss = self.criterion(y_hat, y)
        total_loss = base_loss + extra_loss

        # 查看输入输出维度，仅在整个训练的第一个 epoch 的第一个 batch 打印
        if self.current_epoch == 0 and batch_idx == 0:
            # 获取 y（目标）
            y = batch[-1]  # 目标通常在 batch 的最后一个元素

            # 判断 batch[0] 的结构并打印相应的输入维度
            if isinstance(batch[0], (tuple, list)):
                # 如果 batch[0] 是一个元组或列表，表示输入有多个部分
                for idx, input_tensor in enumerate(batch[0]):
                    if self.trainer.is_global_zero:
                        print(f"\n Training started: Input {idx} shape={input_tensor.shape}")
                if self.trainer.is_global_zero:
                    print(f"Training started: y.shape={y.shape}")
            else:
                # 如果 batch[0] 是一个单一的张量
                if self.trainer.is_global_zero:
                    print(f"\n Training started: x.shape={batch[0].shape}, y.shape={y.shape}")

        self.log('train_base_loss', base_loss.mean())
        self.log('train_total_loss', total_loss.mean())
        self.log('train_extra_loss', extra_loss)

        return total_loss.mean()

    def validation_step(self, batch, batch_idx):
        y_hat, y, _ = self._process_one_batch(batch)
        # print(f"Validation step: y_hat.shape={y_hat.shape}, y.shape={y.shape}")
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        # 如果训练阶段做了归一化，则验证时反归一化
        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)

        y_hat = y_hat.contiguous()
        y = y.contiguous()
        # 更新各项指标（这里假设你使用的是 torchmetrics 指标，或者可以用 sklearn 计算）
        self.valid_MAE(y_hat, y)
        self.valid_MAPE(y_hat, y)
        self.valid_RMSE(y_hat, y)
        self.valid_SMAPE(y_hat, y)
        # .compute() 计算的是整个 epoch 的累积值，而不是当前 batch 的值。
        # on_epoch=True 表示在每个 epoch 结束时记录一次。

        # 追加当前步的输出到列表中
        output = {'y_hat': y_hat, 'y': y}
        self.validation_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        y_hat, y, _ = self._process_one_batch(batch)

        # 如果训练时用了归一化，则测试时做反归一化
        if self.trainer.datamodule.normalize:
            y = self._inverse_transform(y)
            y_hat = self._inverse_transform(y_hat)

        y_hat = y_hat.contiguous()
        y = y.contiguous()

        output = {'y_hat': y_hat, 'y': y}
        self.test_outputs.append(output)

        return output

    def predict_step(self, batch, batch_idx):
        y_hat, y, _ = self._process_one_batch(batch)
        # 如果训练时做了归一化则预测时进行反归一化
        y_hat = self._inverse_transform(y_hat) if self.trainer.datamodule.normalize else y_hat
        self._predict_results.append(y_hat)

        return y_hat

    def on_train_epoch_end(self):
        clean_cache()

    def on_fit_end(self):
        clean_cache()

    def on_validation_epoch_end(self):
        # 如果是 sanity check 阶段，清空列表并返回
        if self.trainer.sanity_checking:
            self.validation_outputs.clear()
            return

        # 解包所有验证步骤的输出
        # 这里我们构造一个列表，其中每个元素是一个二元组 (y_hat, y)
        outputs = [(out['y_hat'], out['y']) for out in self.validation_outputs]
        y_hat_list, y_list = zip(*outputs)
        y_hat_cat = torch.cat(y_hat_list, dim=0)
        y_cat = torch.cat(y_list, dim=0)

        # pred_len = y_hat_cat.shape[1]  # original
        pred_len = self.pred_len

        # if len(y_hat_cat[0].shape) == 2:
        #     preds = y_hat_cat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        #     gt = y_cat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        # else:
        #     preds = y_hat_cat.view(-1, self.nb_flows, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
        #     gt = y_cat.view(-1, self.nb_flows, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()

        # if self.show_fig:
        #     self._vis_gt_preds_topk(gt, preds, step='val', save_flag=True)

        # 清空列表，准备下一次验证周期
        self.validation_outputs.clear()
        clean_cache() # 清理缓存


    def on_test_epoch_end(self):
        """
        在测试周期结束时，聚合所有 test_step 输出，并计算各项指标，
        然后将指标写入日志。
        """
        # 检查是否有测试输出（self.test_outputs 应该在 __init__ 中初始化为空列表）
        if not hasattr(self, 'test_outputs') or len(self.test_outputs) == 0:
            if self.trainer.is_global_zero:
                print("No test outputs collected!")
            return

        # 聚合所有 test_step 输出, 拼接所有样本数，第一维度 B -> B * num_B
        outputs = self.test_outputs  # outputs 是一个列表，每个元素是 dict {'y_hat': ..., 'y': ...}
        y_hat_list = [out['y_hat'] for out in outputs]
        y_list = [out['y'] for out in outputs]
        y_hat_cat = torch.cat(y_hat_list, dim=0)
        y_cat = torch.cat(y_list, dim=0)

        # pred_len = y_hat_cat.shape[1] # original version
        pred_len = self.pred_len
        # 根据 y_hat 的维度处理数据
        # 如果 y_cat 的维度是 [B, N, pred_len]，则这行代码将把维度转换为 [B * N, pred_len, 1]。
        # 如果 y_cat 的维度是 [B, N, pred_len, C]，则这行代码将把维度转换为 [B * N, pred_len, 1, C]。
        if len(y_hat_cat[0].shape) == 2:
            preds = y_hat_cat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
            gt = y_cat.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        else:
            preds = y_hat_cat.view(-1, self.nb_flows, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()
            gt = y_cat.view(-1, self.nb_flows, pred_len, self.trainer.datamodule.n_grids).cpu().detach().numpy()

        # 使用 sklearn 计算指标（注意此处指标计算方式与 torchmetrics 不同）
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

        mae = mean_absolute_error(gt.ravel(), preds.ravel()) # ravel 展平为 1-dim
        mape = mean_absolute_percentage_error(gt.ravel(), preds.ravel())
        r2 = r2_score(gt.ravel(), preds.ravel())
        rmse = np.mean(
            [mean_squared_error(gt[i].flatten(), preds[i].flatten(), squared=False) for i in range(gt.shape[0])])

        # 如果有多个流（例如 nb_flows == 2），计算每个流的指标
        if self.nb_flows == 2:
            mae_c1 = mean_absolute_error(gt[:, 0].ravel(), preds[:, 0].ravel())
            mae_c2 = mean_absolute_error(gt[:, 1].ravel(), preds[:, 1].ravel())
            rmse_c1 = np.mean([mean_squared_error(gt[i, 0].flatten(), preds[i, 0].flatten(), squared=False) for i in
                               range(gt.shape[0])])
            rmse_c2 = np.mean([mean_squared_error(gt[i, 1].flatten(), preds[i, 1].flatten(), squared=False) for i in
                               range(gt.shape[0])])

        # 额外处理：例如对某个时间步的预测进行特殊修改
        # ? 怎么还有这种东西
        # preds[-24] = ((gt[-25] + gt[-26] + gt[-27]) / 3.0) * 2.5

        mae_2 = mean_absolute_error(gt.ravel(), preds.ravel())
        mape_2 = mean_absolute_percentage_error(gt.ravel(), preds.ravel())
        r2_2 = r2_score(gt.ravel(), preds.ravel())
        rmse_2 = np.mean(
            [mean_squared_error(gt[i].flatten(), preds[i].flatten(), squared=False) for i in range(gt.shape[0])])
        rmse_c = mean_squared_error(gt.flatten(), preds.flatten(), squared=False)

        self.test_results_table.add_data(
            self.current_epoch,
            mae, mape, r2, rmse,
            mae_2, mape_2, r2_2, rmse_2,
            rmse_c
        )
        # 将数据存入 `wandb.Table`
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"Test Results": self.test_results_table})

        # **竖排打印测试结果**
        if self.trainer.is_global_zero:
            print("\n=============== Test Results ================")
            columns = self.test_results_table.columns  # 获取列名
            for row in self.test_results_table.data:
                for i, col_name in enumerate(columns):
                    value = row[i]
                    if isinstance(value, (int, float)):  # 数值类型
                        print(f"{col_name:<20}: {value:.3f}")
                    else:  # 其他类型（如字符串）
                        print(f"{col_name:<20}: {value}")
                print("-" * 50)  # 分割线

        self._error_analysis(gt, preds, node_idx=10, save_flag=True)

        if self.show_fig:
            self._vis_gt_preds_topk(gt, preds, step='test', save_flag=True)

        if self.show_intermediate_results:
            """在测试结束后记录模型的中间计算结果（仅使用第一个 batch）"""
            # 获取测试集的第一个 batch
            test_loader = self.trainer.datamodule.test_dataloader()
            first_batch = next(iter(test_loader))  # 取出第一个 batch
            X_first_batch, _ = first_batch  # 假设 batch 是 (输入, 目标)
            # 调用日志函数
            X_first_batch = X_first_batch.to(self.device)
            self._log_intermediate_results(X_first_batch)

        # 清空输出列表，便于下个测试周期收集数据
        self.test_outputs.clear()
        clean_cache()  # 清理缓存

    def on_predict_epoch_end(self, *args, **kwargs):
        # 从 self 中获取预测结果
        results = [self._predict_results]  # 注意：Lightning 原本传入的 results 通常是列表套列表
        save_dir = os.path.join(self.result_dir, str(self._get_logger_version()))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results = torch.cat(results[0])
        # pred_len = results.shape[1]
        pred_len = self.pred_len

        if len(results[0].shape) == 2:
            preds = results.view(-1, self.trainer.datamodule.n_grids, pred_len).transpose(1, 2).cpu().detach().numpy()
        else:
            results = results.view(-1, pred_len, self.trainer.datamodule.n_grids)
            preds = results.cpu().detach().numpy()
        np.savez_compressed(os.path.join(save_dir, "preds.npz"), preds)

        # gt 由整个测试数据中，除去最开始的 self.seq_len 个数据后，再按照每个样本包含 n_grids 个特征来重新排列得到的
        gt = self.trainer.datamodule.milan_test[self.seq_len:].reshape(-1, self.trainer.datamodule.n_grids)

        if self.trainer.datamodule.normalize:
            gt = self.trainer.datamodule.scaler.inverse_transform(gt.reshape(-1, 1)).reshape(gt.shape)
        if pred_len > 1:
            gt = np.stack([gt[i:i + pred_len] for i in range(gt.shape[0] - pred_len + 1)], axis=0)

        # 多gpu时gt是多个gpu结果的拼接，和preds维度不一致，会报错
        self._vis_gt_preds_topk(gt, preds, step='pred', save_flag=True)
        clean_cache()  # 清理缓存


    def on_after_backward(self):
        """ 记录梯度信息到 wandb """
        grad_stats = {}

        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # 使用 .real 来提取实部，确保处理复数张量
                grad = torch.real(grad)  # 处理复数张量

                grad_stats[f"{name}/mean"] = grad.mean().item()
                grad_stats[f"{name}/std"] = grad.std(unbiased=False).item()
                # grad_stats[f"params/{name}/max"] = grad.max().item()
                # grad_stats[f"params/{name}/min"] = grad.min().item()

        # 记录梯度信息
        # 只有当 logger 是 WandbLogger 时，才调用 wandb API
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(grad_stats)
        clean_cache()  # 清理缓存

    def _inverse_transform(self, y):
        # 验证/测试时的反归一化时使用的都是训练数据的归一化器
        yn = y.detach().cpu().numpy() # detach from computation graph
        scaler = self.trainer.datamodule.scaler
        yn = scaler.inverse_transform(yn.reshape(-1, 1)).reshape(yn.shape)
        return torch.from_numpy(yn).cuda()

    def _error_analysis(self, gt, preds, *, node_idx=0, save_flag=False):
        """
        通用误差分析函数，支持 nb_flows == 1 或 2，支持结构化与非结构化数据。
        * 以后的参数只能通过键值对调用
        """
        gt = gt.reshape(gt.shape[0], -1, self.nb_flows)  # [T, N, C]
        preds = preds.reshape(preds.shape[0], -1, self.nb_flows)

        fig = plt.figure(figsize=(20, 10 if self.nb_flows == 1 else 20))

        for i in range(self.nb_flows):
            gt_g = gt[:, node_idx, i]
            preds_g = preds[:, node_idx, i]

            ax = fig.add_subplot(self.nb_flows, 2, 2 * i + 1)
            ax.plot(gt_g, label='Ground Truth', color='green')
            ax.plot(preds_g, label='Prediction', color='blue')
            ax.set_title(f"Flow {i} - Node {node_idx}")
            ax.legend()

            ax = fig.add_subplot(self.nb_flows, 2, 2 * i + 2)
            error = preds_g - gt_g
            ax.bar(np.arange(error.shape[0]), error, color='red')
            ax.set_title(f"Error - Flow {i}")

        fig.suptitle('Node {}'.format(node_idx))
        if save_flag:
            save_dir = os.path.join(self.result_dir, str(self._get_logger_version()))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, 'error_analysis.png'))

    def _log_intermediate_results(self, X):
        """默认情况下，STBase 里不做任何操作"""
        raise NotImplementedError

    def _vis_gt_preds_topk(self, gt, preds, *, step, save_flag=False):
        # 只记录主程序上的，避免并行计算时重复记录
        if not self.trainer.is_global_zero:
            return

        # gt shape: [b, pred_len, N]
        print('[vis] Ground truth shape:', gt.shape)

        # === Step 1: 维度整理 ===
        # 期待的维度是：gt.shape == (b, N, pred_len)
        # 期待的维度是：preds.shape == (b, N, pred_len)
        # 如果有通道维度（形如 B, C, N, pred_len），则选择第一个通道
        if len(gt.shape) == 4:
            gt = gt[:, 0, :, :]  # → (B, N, pred_len)
            preds = preds[:, 0, :, :]
            # print('[vis] Selected first channel: ', gt.shape)
        # 如果数据格式是 (B, pred_len, N)，就转置成 (B, N, pred_len)
        if gt.shape[1] == self.pred_len:
            gt = gt.transpose(0, 2, 1)
            preds = preds.transpose(0, 2, 1)
            # print('[vis] Transposed to (B, N, pred_len): ', gt.shape)

        # === Step 2: 可视化单节点的 top9 和 low9 平均流量 ===
        top9_mae_imglist = []
        low9_mae_imglist = []

        for p in range(min(3, gt.shape[2])):  # 遍历 pred_len
            gt_i = gt[:, :, p]  # [b, N]
            pred_i = preds[:, :, p]  # [b, N]

            mean_values = np.mean(gt_i, axis=0)
            top9ind = np.argsort(mean_values)[-9:][::-1]
            low9ind = np.argsort(mean_values)[:9]

            # Top9
            fig_top, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i, idx in enumerate(top9ind):
                ax = axes[i // 3, i % 3]
                ax.plot(gt_i[:, idx], label="gt")
                ax.plot(pred_i[:, idx], label="pred")
                ax.set_title(f"{idx}: MAE={mean_absolute_error(gt_i[:, idx], pred_i[:, idx]):.4f}")
                ax.legend()
            top9_mae_imglist.append(fig_top)

            # Low9
            fig_low, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i, idx in enumerate(low9ind):
                ax = axes[i // 3, i % 3]
                ax.plot(gt_i[:, idx], label="gt")
                ax.plot(pred_i[:, idx], label="pred")
                ax.set_title(f"{idx}: MAE={mean_absolute_error(gt_i[:, idx], pred_i[:, idx]):.4f}")
                ax.legend()
            low9_mae_imglist.append(fig_low)

        # === Step 4: 保存 ===
        if save_flag:
            save_dir = os.path.join(self.result_dir, str(self._get_logger_version()))
            os.makedirs(save_dir, exist_ok=True)
            for i, fig in enumerate(top9_mae_imglist):
                fig.savefig(os.path.join(save_dir, f"top9_{i}.png"))
            for i, fig in enumerate(low9_mae_imglist):
                fig.savefig(os.path.join(save_dir, f"low9_{i}.png"))
            # fig_grid.savefig(os.path.join(save_dir, "top9_mae_grid.png"))

        # === Step 5: wandb 上传 ===
        if isinstance(self.logger, WandbLogger):
            # 确保 step 是合法 key
            safe_step = re.sub(r"[^\w\-\.]", "_", str(step))
            for i, fig in enumerate(top9_mae_imglist):
                self.logger.log_image(key=f"{safe_step}_top9_{i}", images=[wandb.Image(fig)])
            for i, fig in enumerate(low9_mae_imglist):
                self.logger.log_image(key=f"{safe_step}_low9_{i}", images=[wandb.Image(fig)])
            # self.logger.log_image(key=f"{safe_step}_grid", images=[wandb.Image(fig_grid)])

        plt.close('all')


################ old version ##############

    # def _vis_gt_preds_topk(self, gt, preds, *, step, save_flag=False):
    #
    #     if self.trainer.is_global_zero:
    #         print('visual: gt.shape: ', gt.shape)
    #
    #     # scope:  (851, 1, 32, 400) -> (851, 400, 32)
    #     if gt.shape[1] < 5 and gt.shape[2] == self.pred_len:
    #         gt = gt[:,0,:,:].squeeze().transpose(0, 2, 1)
    #         preds = preds[:,0,:,:].squeeze().transpose(0, 2, 1)
    #         if self.trainer.is_global_zero:
    #             print('visual: reshaped gt.shape: ', gt.shape)
    #
    #     pic_number = 3
    #     top9_mae_imglist = []
    #     low9_mae_imglist = []
    #     for p in range(pic_number):
    #         gt_i = gt[:, p, :] # mywat: (851, 400), work
    #         pred_i = preds[:, p, :] # mywat: (851, 400), work
    #         top9_mean_fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    #         top9ind = np.argpartition(np.mean(gt_i, axis=0), -9)[-9:]
    #         for i in range(9):
    #             axes[i // 3, i % 3].plot(gt_i[:, top9ind[i]], label="gt")
    #             axes[i // 3, i % 3].plot(pred_i[:, top9ind[i]], label="pred")
    #             axes[i // 3, i % 3].set_title(f"{top9ind[i]}: MAE: {mean_absolute_error(pred_i[:, top9ind[i]], gt_i[:, top9ind[i]]):9.4f}")
    #             axes[i // 3, i % 3].legend()
    #         top9_mae_imglist.append(top9_mean_fig)
    #
    #         low9_mean_fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    #         low9ind = np.argpartition(np.mean(gt_i, axis=0), 9)[:9]
    #         for i in range(9):
    #             axes[i // 3, i % 3].plot(gt_i[:, low9ind[i]], label="gt")
    #             axes[i // 3, i % 3].plot(pred_i[:, low9ind[i]], label="pred")
    #             axes[i // 3, i % 3].set_title(f"{low9ind[i]}: MAE: {mean_absolute_error(pred_i[:, low9ind[i]], gt_i[:, low9ind[i]]):9.4f}")
    #             axes[i // 3, i % 3].legend()
    #         low9_mae_imglist.append(low9_mean_fig)
    #
    #     gt_flatten = gt.reshape(-1, self.trainer.datamodule.n_grids)
    #     pred_flatten = preds.reshape(-1, self.trainer.datamodule.n_grids)
    #     top9_mae_grid = plt.figure(figsize=(30, 15))
    #     top9_mae_indexes = np.argpartition(np.sum(np.absolute(gt_flatten-pred_flatten), axis=1), -9)[-9:]
    #     outer = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)
    #     rows, cols = self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols
    #     for i in range(9):
    #         inner = gridspec.GridSpecFromSubplotSpec(1, 2,
    #                 subplot_spec=outer[i], wspace=0.2, hspace=0.1)
    #         _x = np.arange(rows)
    #         _y = np.arange(cols)
    #         _xx, _yy = np.meshgrid(_x, _y)
    #         x, y = _xx.ravel(), _yy.ravel()
    #         z_gt = gt_flatten[top9_mae_indexes[i], :]
    #         z_pred = pred_flatten[top9_mae_indexes[i], :]
    #         bottom = np.zeros_like(z_gt)
    #
    #         cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    #         max_height = np.max((z_gt, z_pred))   # get range of colorbars so we can normalize
    #         min_height = np.min((z_gt, z_pred))
    #
    #         # scale each z to [0,1], and get their rgb values
    #         rgb1 = [cmap((k-min_height)/max_height) for k in z_pred]
    #         rgb2 = [cmap((k-min_height)/max_height) for k in z_gt]
    #
    #         ax1 = top9_mae_grid.add_subplot(inner[0], projection='3d')
    #         ax2 = top9_mae_grid.add_subplot(inner[1], projection='3d', sharez=ax1, sharex=ax1, sharey=ax1)
    #         ax1.bar3d(x, y, bottom, 1, 1, z_pred, color=rgb1, shade=True)
    #         ax2.bar3d(x, y, bottom, 1, 1, z_gt, color=rgb2, shade=True)
    #         ax1.set_title(f"mae: {mean_absolute_error(z_pred, z_gt):9.4f} - index: {top9_mae_indexes[i]}")
    #
    #     if save_flag:
    #         save_dir = os.path.join(self.result_dir, str(self._get_logger_version()))
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         [img.savefig(os.path.join(save_dir, f"preds_top9_{i}.png")) for i, img in enumerate(top9_mae_imglist)]
    #         [img.savefig(os.path.join(save_dir, f"preds_low9_{i}.png")) for i, img in enumerate(low9_mae_imglist)]
    #         top9_mae_grid.savefig(os.path.join(save_dir, "preds_top9_mae_grid.png"))
    #
    #     if isinstance(self.logger, WandbLogger):
    #         for i, img in enumerate(top9_mae_imglist):
    #             self.logger.log_image(key=f'{step}_top9_{i}', images=[wandb.Image(img)])
    #         for i, img in enumerate(low9_mae_imglist):
    #             self.logger.log_image(key=f'{step}_low9_{i}', images=[wandb.Image(img)])
    #         self.logger.log_image(key=f'{step}_grid', images=[wandb.Image(top9_mae_grid)])
    #
    #     plt.close('all')


 # === Step 3: 3D Top9 MAE 网格误差图 ===
        # T, N, P = gt.shape
        # gt_flat = gt.reshape(T, N * P)
        # pred_flat = preds.reshape(T, N * P)
        # total_mae = np.mean(np.abs(gt_flat - pred_flat), axis=1)
        # top_indices = np.argsort(total_mae)[-9:][::-1]
        #
        # fig_grid = plt.figure(figsize=(30, 15))
        # outer = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)
        # rows, cols = self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols
        # for i, idx in enumerate(top_indices):
        #     gt_i = gt_flat[idx].reshape(rows, cols)
        #     pred_i = pred_flat[idx].reshape(rows, cols)
        #     bottom = np.zeros_like(gt_i)
        #
        #     x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        #     x, y = x.ravel(), y.ravel()
        #     z_gt = gt_i.ravel()
        #     z_pred = pred_i.ravel()
        #
        #     max_val = max(z_gt.max(), z_pred.max())
        #     min_val = min(z_gt.min(), z_pred.min())
        #
        #     norm = lambda z: (z - min_val) / (max_val - min_val + 1e-6)
        #     cmap = cm.get_cmap('jet')
        #     rgb_gt = [cmap(val) for val in norm(z_gt)]
        #     rgb_pred = [cmap(val) for val in norm(z_pred)]
        #
        #     inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i])
        #     ax1 = fig_grid.add_subplot(inner[0], projection='3d')
        #     ax2 = fig_grid.add_subplot(inner[1], projection='3d')
        #     ax1.bar3d(x, y, 0, 1, 1, z_pred, color=rgb_pred, shade=True)
        #     ax2.bar3d(x, y, 0, 1, 1, z_gt, color=rgb_gt, shade=True)
        #     ax1.set_title(f"Pred MAE={mean_absolute_error(z_gt, z_pred):.4f}")


# def _error_analysis(self, gt, preds, *, node_idx = 10 ,grid=(10, 10), save_flag=False):

# if self.nb_flows == 1:
#     fig = plt.figure(figsize=(20, 10))
#     gt = gt.reshape(-1, self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols)
#     preds = preds.reshape(-1, self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols)
#     gt_g = gt[:, grid[0], grid[1]]
#     preds_g = preds[:, grid[0], grid[1]]
#     ax = fig.add_subplot(1, 2, 1)
#     ax.plot(gt_g, label='Ground Truth', color='green')
#     ax.plot(preds_g, label='Prediction', color='blue')
#     ax.legend()
#
#     ax = fig.add_subplot(1, 2, 2)
#     error = preds_g - gt_g
#     ax.bar(np.arange(error.shape[0]), error, color='red')

# if self.nb_flows == 1:
#     fig = plt.figure(figsize=(20, 10))
#     gt = gt.reshape(gt.shape[0], -1)  # [T, N]
#     preds = preds.reshape(preds.shape[0], -1)
#
#     gt_node = gt[:, node_idx]
#     preds_node = preds[:, node_idx]
#
#     ax = fig.add_subplot(1, 2, 1)
#     ax.plot(gt_node, label='Ground Truth', color='green')
#     ax.plot(preds_node, label='Prediction', color='blue')
#     ax.legend()
#
#     ax = fig.add_subplot(1, 2, 2)
#     error = preds_node - gt_node
#     ax.bar(np.arange(error.shape[0]), error, color='red')
#
# elif self.nb_flows == 2:
#     fig = plt.figure(figsize=(20, 20))
#     gt = gt.reshape(-1, self.nb_flows, self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols)
#     preds = preds.reshape(-1, self.nb_flows, self.trainer.datamodule.n_rows, self.trainer.datamodule.n_cols)
#     for i in range(2):
#         gt_g = gt[:, i, grid[0], grid[1]]
#         preds_g = preds[:, i, grid[0], grid[1]]
#         ax = fig.add_subplot(self.nb_flows, 2, 2*i+1)
#         ax.plot(gt_g, label='Ground Truth', color='green')
#         ax.plot(preds_g, label='Prediction', color='blue')
#         ax.legend()
#
#         ax = fig.add_subplot(self.nb_flows, 2, 2*i+2)
#         error = preds_g - gt_g
#         ax.bar(np.arange(error.shape[0]), error, color='red')