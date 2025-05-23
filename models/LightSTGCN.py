import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.io.planetoid import edge_index_from_dict
from models.STBase import STBase
import networkx as nx
import os
import pandas as pd
from utils.funcs import *
from scipy.sparse.linalg import eigs
from utils.registry import register

@register("LightSTGCN")
class LightSTGCN(STBase):
    def __init__(self,
                 close_len: int = 6,
                 pred_len: int = 3,
                 feature_len: int = 64,
                 cheb_k: int = 3,
                 period_len: int = 0,
                 trend_len: int = 0,
                 grid_range: tuple = (0, 2, 0, 6),
                 *args, **kwargs):
        super(LightSTGCN, self).__init__(*args, **kwargs)

        self.pred_len = pred_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.seq_len = close_len + period_len + trend_len

        # 用网格图，k_average = 3.8
        # adj_mx = np.load("/data/scratch/jiayin/Adj_SmoothGL_Milan10Min_Internet.npy")
        # AdjKnn2_D15_Milan10Min_Internet.npy
        # AdjKnn3_D29_Milan10Min_Internet.npy
        # AdjKnn4_D45_Milan10Min_Internet.npy
        # AdjKnn5_D59_Milan10Min_Internet.npy
        self.n_row = grid_range[1] - grid_range[0] + 1
        self.n_col = grid_range[3] - grid_range[2] + 1
        G = nx.grid_2d_graph(self.n_row, self.n_col)
        adj_mx = nx.adjacency_matrix(G)
        # adj_mx = pd.read_csv("/home/jiayin/PycharmProjects/Spatio-Temporal-Analysis-Telecom-Italia/experiments/experiments/results/GLLowPass_04132226/A.csv").to_numpy()
        self.edge_index = Adj2EdgeList(adj_mx)

        # Chebyshev Graph Convolution (ChebConv)
        self.cheb_conv = cheb_conv(in_channels=1, out_channels = 64,
                                  K = cheb_k, Adj=adj_mx)  # Chebyshev图卷积

        # Time Convolution (Conv2d for (N,T) convolution)
        self.time_conv = nn.Conv2d(in_channels = 64, out_channels = feature_len,
                                   kernel_size=(1, 3),
                                   stride=(1, 1), padding=(0, 1))  # 时间卷积

        # Residual Convolution (Conv2d for residual connection)
        self.residual_conv = nn.Conv2d(in_channels = 1, out_channels = feature_len,
                                       kernel_size=(1, 1))  # 残差卷积

        # Layer Normalization
        self.ln = nn.LayerNorm(feature_len)  # 对F_out维度做层归一化

        # Final Convolution (restoring F_in back)
        self.final_conv = nn.Conv2d(in_channels=close_len, out_channels=pred_len,
                                    kernel_size=(1, feature_len))  # 对F维度做卷积
        # Initialize parameters
        self.save_hyperparameters()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):

        # 如果是列表输入（多组件输入）
        if isinstance(x, list):
            xc = x[0]
            xp = x[1] if len(x) > 1 else None
            xt = x[2] if len(x) > 2 else None
        else:
            xc = x
            xp = None
            xt = None

        x = xc
        # print(f"[DEBUG] shape(x): {x.shape}")
        # torch.Size([32, 12, 1, 3, 7])

        # (b, N, F_in, T)
        x_ori = x.permute(0, 2, 1, 3) # (b, F_in, N, T)

        # Step 1: Apply Chebyshev Convolution (ChebConv)
        x = self.cheb_conv(x)  # (b, N, F_out, T)

        # Step 2: Apply Time Convolution (Conv2d for T dimension)
        x = x.permute(0, 2, 1, 3)  # (b, F_out, N, T)
        x = self.time_conv(x)  # (b, F_out, N, T)

        # Step 3: Residual Connection
        residual = self.residual_conv(x_ori)  # (b, F_out, N, T)
        x = x + residual  # Add residual connection (b, F_out, N, T)

        # Step 4: Apply Layer Normalization
        x = x.permute(0, 2, 3, 1) # (b, N, T, F_out)
        x = self.ln(x)  # (b, N, T, F_out)

        # Step 5: Final Convolution (to restore F_in)
        x = x.permute(0, 2, 1, 3)  # (b, T, N, F_out)
        x = self.final_conv(x)  # (b, L, N, F_in)
        x = x.permute(0, 1, 3, 2) # (b, L, F_in, N)
        # x = x.resahpe(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        # y.shape = torch.Size([32, 32, 1, 20, 20])

        return x

class cheb_conv(nn.Module):

    def __init__(self, K, Adj, in_channels, out_channels):
        super(cheb_conv, self).__init__()
        self.K = K
        self.in_channels = in_channels # 固定是1，输入的每个节点的特征维度
        self.out_channels = out_channels # nb_chev_filter, 默认是64，输出的每个节点的特征维度
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        L_tilde = scaled_Laplacian(Adj)
        self.register_buffer("cheb_polynomials", torch.from_numpy(cheb_polynomial(L_tilde, K)))

    def forward(self, x):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps): # 对每个时间步单独算卷积，怪不得慢
            graph_signal = x[:, :, :, time_step]  # Extract features for the current time step (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).type_as(x)  # (b, N, F_out)
            for k in range(self.K): # 对每一阶chev多项式
                # chev 多项式 Tk (N, N), 由 L_tilde 矩阵计算得到
                # L_tilde ：网格图结构 + 特征值归一化
                T_k = self.cheb_polynomials[k]
                # chev 多项式 Tk (N, N) * spatial_attention (b, N, N) -> (b, N, N)
                # (in_channels, out_channels)
                theta_k = self.Theta[k]
                # (b, N, N) @ (b, N, F_in) -> (b, N, F_in)
                rhs = T_k.permute(1,0).matmul(graph_signal)
                # (b, N, F_in) @ (F_in, F_out) -> (b, N, F_out)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))  # Add a time dimension (b, N, F_out, 1)

        # Concatenate over time and apply ReLU: (b, N, F_out, T)
        out = F.relu(torch.cat(outputs, dim=-1))
        return out

def scaled_Laplacian(W):

    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1)) # 因为用的是 grid_graph, 故不会有孤立节点问题
    L = D - W
    L = L.astype(np.float32)

    # Compute the largest eigenvalue for scaling
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):

    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), np.asarray(L_tilde)]
    for i in range(2, K):
        cheb_polynomials.append(np.asarray(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]))
    cheb_polynomials = np.stack(cheb_polynomials, axis=0).astype(np.float32)
    return cheb_polynomials

if __name__ == '__main__':
    model = LightSTGCN(close_len=6, pred_len=3, feature_len=64, cheb_k=3)
    x = torch.randn(32, 21, 1, 6)
    y = model(x)
    print(y.shape)
    # torch.Size([32, 3, 1, 21]), [B, pred_len, 1, N_all]
