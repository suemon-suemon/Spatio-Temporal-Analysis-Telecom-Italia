import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.STBase import STBase

# best result:
# alpha 0.002 beta 0.002
# ETTh1 480 -> 720 M24 num4 0.412  192 M2 num8 0.384    336 0.402

# Residual layer
# self.residual_layer = torch.nn.Sequential(
#     torch.nn.Linear(self.seq_len, self.hid_dim),
#     torch.nn.ReLU(),
#     torch.nn.Linear(self.hid_dim, self.pred_len),
# )
# # Initialize residual layers
# for param in self.residual_layer.parameters():
#     if param.dim() > 1:
#         nn.init.kaiming_normal_(param)

class SCOPE(STBase):
    def __init__(self,
                 num_nodes: int = 400, # 节点数目
                 time_basis_number: int = 4,  # 基的数量, = pattern_num
                 downsampling_rate: int = 24,  # 降采样率, = M
                 close_len: int = 128,  # 历史序列长度, seq_len
                 pred_len: int = 64,  # 预测序列长度
                 enc_in: int = 1,  # 通道数，即业务数
                 individual: bool = False,  # 是否通道独立，即每个通道都有不同的pattern_num个基
                 complex_num: bool = True,  # 是否使用复数基
                 alpha: float = 0.002,  # 正交约束损失系数
                 beta: float = 0.002,  # 重构损失系数
                 *args, **kwargs):
        super(SCOPE, self).__init__(*args, **kwargs)

        self.num_nodes = num_nodes
        self.complex_num = complex_num
        self.M = downsampling_rate
        self.seq_len = close_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.pattern_num = time_basis_number
        self.individual = individual
        self.alpha = alpha
        self.beta = beta
        self.recon_loss_criteria = nn.MSELoss()
        self.hid_dim = 100

        self.pattern_num_original = 100
        self.cut_freq = 180
        self.gamma = 0.005
        self.before_state = True
        self.is_prune = False
        self.pruned_num = 0

        # Pattern dictionary (基字典)
        if self.individual:
            self.pattern_dict_list = nn.ModuleList()
            for i in range(self.enc_in):
                embedding = nn.Embedding(self.pattern_num, (self.seq_len + self.pred_len) // self.M)
                if self.complex_num:  # Convert to complex
                    embedding.weight.data = embedding.weight.data.to(torch.cfloat)
                nn.init.kaiming_normal_(embedding.weight)
                self.pattern_dict_list.append(embedding)
        else:
            self.pattern_dict = nn.Embedding(self.pattern_num, (self.seq_len + self.pred_len) // self.M)
            if self.complex_num:
                self.pattern_dict.weight.data = self.pattern_dict.weight.data.to(torch.cfloat)
            nn.init.kaiming_normal_(self.pattern_dict.weight)

        # Sub-sequence mixing layer
        self.inter_sequence_mix = nn.Linear(self.M, self.M, bias=False)

        # Coefficient compensation network
        if self.individual:
            self.corr_shift_layer_module = nn.ModuleList()
            for i in range(self.enc_in):
                if self.complex_num:
                    self.corr_shift_layer_module.append(nn.Linear(self.pattern_num, self.pattern_num, bias=False).to(torch.cfloat))
                else:
                    self.corr_shift_layer_module.append(nn.Linear(self.pattern_num, self.pattern_num, bias=False))
                nn.init.kaiming_normal_(self.corr_shift_layer_module[i].weight)
        else:
            if self.complex_num:
                self.corr_shift_layer = nn.Linear(self.pattern_num, self.pattern_num, bias=False).to(torch.cfloat)
            else:
                self.corr_shift_layer = nn.Linear(self.pattern_num, self.pattern_num, bias=False)
            nn.init.kaiming_normal_(self.corr_shift_layer.weight)

        # Buffers for adaptive pruning (unused but don't change)
        self.register_buffer('corr_buffer', None)
        self.buffer_size = 100000
        self.similarity_threshold = 0.70

    def _compute_forward_results(self, x):
        """
        前向传播函数 (Forward)
        参数:
            x: 形状 [batch_size, num_nodes, seq_len, enc_in]
               - batch_size: 一个 batch 中序列样本个数
               - num_nodes: 节点数，通常是多个区域或不同传感器
               - seq_len: 每条时序数据的长度 (时间步数)
               - enc_in: 每个时间步的特征/通道数
        返回:
            y: 形状 [batch_size, num_nodes, pred_len, enc_in]
               - 预测输出, 时间步数 = pred_len, 特征数 = enc_in
        """

        # 输入数据的形状: [batch_size, num_nodes, seq_len, enc_in]
        batch_size, num_nodes, seq_len, enc_in = x.shape

        # 交换维度, 变为 [batch_size, num_nodes, enc_in, seq_len]
        x_reshape = x.permute(0, 1, 3, 2)  # => [batch_size, num_nodes, enc_in, seq_len]

        # 降采样 => [batch_size, num_nodes, enc_in, seq_len/M, M]
        x_downsampling = x_reshape.reshape(
            batch_size, num_nodes, self.enc_in, self.seq_len // self.M, self.M
        )

        # 子序列混合, 维度不变 => [batch_size, num_nodes, enc_in, seq_len/M, M]
        x_downsampling_mixed = self.inter_sequence_mix(x_downsampling)

        # 使用 pattern 字典计算相关系数
        if self.individual:
            # ------------------------- individual 模式: 每个通道单独处理 -------------------------
            # 先准备一个空张量接收结果
            # 形状: [batch_size, num_nodes, enc_in, pattern_num, M]
            tmp = torch.zeros(
                batch_size, num_nodes, enc_in, self.pattern_num, self.M,
                device=x.device, dtype=torch.cfloat
            )

            for i in range(self.enc_in):
                tmp[:, :, i] = torch.einsum(
                    'bslm,nl->bsnm', # s = num_nodes
                    x_downsampling_mixed[:, :, i].to(torch.cfloat),
                    torch.conj(self.pattern_dict_list[i].weight[:, :self.seq_len // self.M])
                )
            corr_before_com = tmp

            # 用 shift layer 修正系数
            corr_after_com = torch.zeros_like(corr_before_com)
            for i in range(self.enc_in):
                corr_after_com[:, :, i] = self.corr_shift_layer_module[i](
                    corr_before_com.permute(0, 1, 2, 4, 3)[:, :, i]
                ).permute(0, 2, 1) # 这里维度不对，用到再调吧

            # 与 pattern dict 做内积，得到预测模式
            tmp_pattern = torch.zeros(
                batch_size, num_nodes, self.enc_in, (self.seq_len + self.pred_len) // self.M, self.M,
                device=x.device, dtype=torch.cfloat
            )

            for i in range(self.enc_in):
                tmp_pattern[:, :, i] = torch.einsum(
                    'bsnm,nl->bslm', # s = num_nodes
                    corr_after_com[:, :, i],
                    self.pattern_dict_list[i].weight
                )
            x_recon_longer = tmp_pattern

        else:
            # ------------------------- shared 模式: 所有通道共用一个 pattern dict -------------------------
            if self.complex_num:
                corr_before_com = torch.einsum(
                    'bsclm,nl->bscnm', # s = num_nodes
                    x_downsampling_mixed.to(torch.cfloat),
                    torch.conj(self.pattern_dict.weight[:, :self.seq_len // self.M])
                )
            else:
                corr_before_com = torch.einsum(
                    'bsclm,nl->bscnm', # s = num_nodes
                    x_downsampling_mixed,
                    self.pattern_dict.weight[:, :self.seq_len // self.M]
                )
            # [batch_size, num_nodes, enc_in, M, K]
            corr_after_com = self.corr_shift_layer(corr_before_com.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)

            x_recon_longer = torch.einsum(
                'bscnm,nl->bsclm', # s = num_nodes
                corr_after_com,
                self.pattern_dict.weight
            )

        # 还原并返回最终的预测结果
        x_recon_longer = x_recon_longer.reshape(batch_size, num_nodes, self.enc_in, self.seq_len + self.pred_len)
        # 取实部作为信号恢复值，因为虚部代表相位，可以舍弃
        x_recon_longer = x_recon_longer.real

        # => [batch_size, num_nodes, seq_len, enc_in]
        x_recon = x_recon_longer[:, :, :, :self.seq_len].permute(0, 1, 3, 2)
        # => [batch_size, num_nodes, pred_len, enc_in]
        y = x_recon_longer[:, :, :, -self.pred_len:].permute(0, 1, 3, 2)

        # 重构损失
        loss_recon = self.recon_loss_criteria(torch.abs(x_recon), torch.abs(x)) * self.beta
        # 正交损失
        loss_orth = self.orthogonal_regularization() * self.alpha

        return {
            "corr_before_com": corr_before_com,
            "corr_after_com": corr_after_com,
            "x_recon": x_recon,
            "y": y,
            "extra_loss": loss_recon + loss_orth,
        }


    def forward(self, x, skip=True):
        """前向传播调用 `_compute_forward_results`"""
        results = self._compute_forward_results(x)
        return results["y"], results["extra_loss"]

    def _log_intermediate_results(self, X):
        """记录模型的中间结果到本地"""
        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        X = X.to(self.device)
        results = self._compute_forward_results(X)
        # wandb_logger = self.logger.experiment  # 使用 Lightning 传入的 wandb

        corr_before = results["corr_before_com"].detach().cpu().numpy()
        corr_after = results["corr_after_com"].detach().cpu().numpy()
        np.save(os.path.join(save_dir, "corr_before.npy"), corr_before) # 将 corr 存储为 .npy 文件
        np.save(os.path.join(save_dir, "corr_after.npy"), corr_after)

        if not self.individual:
            Dic_learned = self.pattern_dict.weight.detach().cpu().numpy()
            np.save(os.path.join(save_dir, "D_learned.npy"), Dic_learned)

        # 画出字典矩阵图像并存储
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
        rows_to_plot = [0, int(self.pattern_num//2), self.pattern_num-1]
        for i, row in enumerate(rows_to_plot):
            # 获取时序数据
            time_series = Dic_learned[row, 1:128].real
            # 计算 FFT 变换
            fft_values = np.fft.fft(time_series)  # 计算 FFT
            fft_magnitudes = np.abs(fft_values)  # 计算振幅
            fft_frequencies = np.fft.fftfreq(len(time_series))  # 计算频率轴
            # 画时序信号（左列）
            axes[i, 0].plot(time_series, label=f'D Row {row}')
            axes[i, 0].grid(True)
            axes[i, 0].set_title(f'Time Series - Row {row}')
            axes[i, 0].legend()
            # 画频域信号（右列）
            axes[i, 1].plot(fft_frequencies[:len(fft_frequencies) // 2], fft_magnitudes[:len(fft_magnitudes) // 2],
                            label=f'FFT Row {row}')
            axes[i, 1].grid(True)
            axes[i, 1].set_title(f'FFT - Row {row}')
            axes[i, 1].legend()
        plt.tight_layout()
        imag_name = "D_time_freq.png"
        fig.savefig(os.path.join(save_dir, imag_name), format='png')
        plt.close()

        # 画出C
        fig_C, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
        # [batch_size, num_nodes, enc_in, M, K]
        rows_to_plot = [0, int(self.M//2), self.M-1]
        for i, row in enumerate(rows_to_plot):
            # 绘制 CorrBefore
            axes[i, 0].plot(corr_before[0, 0, 0, :, row].real, label=f'series {row}',
                            marker='o', linestyle='None')
            # 添加竖直线
            for j, val in enumerate(corr_before[0, 0, 0, :, row].real):
                axes[i, 0].vlines(j, 0, val, color='b', alpha=0.5)  # 竖直线
            axes[i, 0].grid(True)
            axes[i, 0].set_title(f'Before: Downsampled series {row}')
            axes[i, 0].legend()

            # 绘制 CorrAfter
            axes[i, 1].plot(corr_after[0, 0, 0, :, row].real, label=f'series {row}',
                            marker='o', linestyle='None', color='g')
            # 添加竖直线
            for j, val in enumerate(corr_after[0, 0, 0, :, row].real):
                axes[i, 1].vlines(j, 0, val, color='g', alpha=0.5)  # 竖直线
            axes[i, 1].grid(True)
            axes[i, 1].set_title(f'After: Downsampled series {row}')
            axes[i, 1].legend()
        plt.tight_layout()
        imag_name_C = "C_before_after_.png"
        fig_C.savefig(os.path.join(save_dir, imag_name_C), format='png')
        plt.close()

        # 正交约束损失
    def orthogonal_regularization(self):
        if self.individual:
            pattern_stack = torch.stack([pattern_dict.weight for pattern_dict in self.pattern_dict_list],
                                        dim=0)  # [enc_in, pattern_num, len]
            pattern = torch.einsum('ipm,iqm->ipq', pattern_stack,
                                   torch.conj(pattern_stack))  # [enc_in, pattern_num, pattern_num]
            pattern = torch.mean(pattern, dim=0)  # [pattern_num, pattern_num]
            loss = torch.norm(pattern - torch.eye(pattern.shape[0], device=pattern.device), p='fro')
        else:
            pattern = self.pattern_dict.weight
            pattern = pattern @ torch.conj(pattern.T)
            loss = torch.norm(pattern - torch.eye(pattern.shape[0], device=pattern.device), p='fro')
        return loss

    # # 重构损失
    # def recon_loss(self, x_recon, x):
    #     return self.recon_loss_criteria(x_recon, x) * self.beta

    # 测试用，以下不用管
    def _update_corr_buffer(self, corr):
        """
        Update the correlation buffer with new correlation magnitudes.
        """
        # Compute magnitude
        corr_magnitude = corr.abs()  # [batch_size, enc_in, pattern_num]

        if self.corr_buffer is None:
            self.corr_buffer = corr_magnitude
        else:
            if corr.size(0) != self.corr_buffer.size(0):
                self.corr_buffer = corr_magnitude
            else:
                self.corr_buffer = torch.cat([self.corr_buffer, corr_magnitude], dim=0)
                if self.corr_buffer.shape[0] > self.buffer_size:
                    self.corr_buffer = self.corr_buffer[-self.buffer_size:]

    def _calculate_mean_corr_magnitude(self):
        """
        Calculate the mean correlation magnitude for each pattern.
        """
        if self.corr_buffer is None:
            raise ValueError("Correlation buffer is empty. Need to collect correlations before pruning.")

        # Mean over buffer
        mean_corr = torch.mean(self.corr_buffer, dim=0)  # [enc_in, pattern_num] if individual else [pattern_num]
        # if self.individual:
        #     mean_corr = torch.mean(mean_corr, dim=0)  # [pattern_num]
        # else:
        mean_corr = torch.mean(mean_corr, dim=0)  # Single mean if shared
        return mean_corr

    def corr_norm(self):
        # """
        # Normalize the correlation matrix.
        # """
        # if self.corr_buffer is None:
        #     raise ValueError("Correlation buffer is empty. Need to collect correlations before pruning.")

        # Normalize over buffer
        corr_norm = torch.mean(self.corr.abs()) * self.gamma
        return corr_norm


    def adaptive_prune(self):
        """
        Perform adaptive pruning on the pattern dictionaries based on correlation magnitudes
        and pattern orthogonality.
        """
        with torch.no_grad():
            mean_corr = self._calculate_mean_corr_magnitude()  # [pattern_num] or scalar
            mean_corr_sorted = torch.sort(mean_corr).values
            plt.figure()
            plt.stem(mean_corr_sorted.cpu().numpy())
            plt.show()

            sorted_indices = torch.argsort(mean_corr)

            # Determine threshold (e.g., remove bottom 30%)
            prune_ratio = 0.0
            num_prune = int(self.pattern_num_original * prune_ratio)
            num_prune = num_prune - self.pruned_num
            prune_candidates = sorted_indices[:num_prune]
            print(prune_candidates)
            # Compute pairwise similarity
            patterns = self.pattern_dict.weight  # [pattern_num, len]
            patterns_abs = patterns.abs()
            similarity = torch.mm(patterns_abs[prune_candidates], patterns_abs.T) / (
                    torch.norm(patterns_abs[prune_candidates], dim=1, keepdim=True) @ torch.norm(patterns_abs, dim=1,
                                                     keepdim=True).T)

            # Initialize set to prune
            to_prune = set(prune_candidates.tolist())
            prune_list = []

            for i, idx in enumerate(prune_candidates):
                # print(idx.item())
                sim = similarity[i]
                similar_idxs = torch.where(sim > self.similarity_threshold)[0]
                for s in similar_idxs:
                    s_idx = s.item()
                    if s_idx not in prune_list and s_idx != idx.item():
                        prune_list.append(idx.item())

            to_prune = set(prune_list)


            pruned_indices = sorted(to_prune)

            # Prune the shared embedding
            kept_indices = [idx for idx in range(self.pattern_num) if idx not in pruned_indices]
            if len(kept_indices) == 0:
                raise ValueError("All patterns are pruned. Adjust pruning ratio or similarity threshold.")
            new_weight = self.pattern_dict.weight.data[kept_indices]
            # new_weight = nn.Parameter(new_weight).to(torch.cfloat)
            new_embedding = nn.Embedding(len(kept_indices), self.seq_len + self.pred_len).to(torch.cfloat)
            new_embedding.weight = nn.Parameter(new_weight)

            # print(new_embedding.weight.requires_grad)
            self.pattern_dict = new_embedding
            print(self.pattern_dict.weight.requires_grad)
            self.pattern_num = len(kept_indices)
            self.pruned_num = self.pattern_num_original - self.pattern_num
            print("Pruned patterns:", self.pruned_num, "Remaining patterns:", self.pattern_num)

