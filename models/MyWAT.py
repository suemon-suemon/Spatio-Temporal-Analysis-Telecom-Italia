import io
from PIL import Image
import os
import networkx as nx
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
from utils.funcs import *
from models.modules import *
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import remove_self_loops, add_self_loops
import matplotlib.pyplot as plt
from models.STBase import STBase
from scipy.sparse.linalg import eigs


def sin_round(x):
    # 近似 round 的连续函数
    return x - torch.sin(2 * np.pi * x) / (2 * np.pi)

# 自定义操作类，包含前向传播和反向传播
class SinRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 记录输入张量，方便反向传播时使用
        ctx.save_for_backward(input)
        # 将输入转换为整数，这里可以使用 round() 来做取整
        return torch.round(input).long().item()

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中取出输入张量
        input, = ctx.saved_tensors
        # 使用 sin_round 作为梯度近似函数
        grad_input = grad_output * (1 - torch.cos(2 * np.pi * input))
        return grad_input


class cheb_conv_withSAt(nn.Module):
    '''
    K-order Chebyshev graph convolution with spatial attention.

    Parameters
    ----------
    K: int
        The order of the Chebyshev polynomial.
    in_channels: int
        The number of input channels (features per vertex).
    out_channels: int
        The number of output channels (features per vertex).
    L_tilde: torch.Tensor
        The scaled Laplacian matrix of the graph.

    Attributes
    ----------
    Theta: torch.nn.ParameterList
        List of learnable parameters for each Chebyshev filter.
    cheb_polynomials: torch.Tensor
        Precomputed Chebyshev polynomials of order K.
    '''

    def __init__(self, K, in_channels, out_channels, L_tilde):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.in_channels = in_channels # 固定是1，输入的每个节点的特征维度
        self.out_channels = out_channels # nb_chev_filter, 默认是64，输出的每个节点的特征维度
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.register_buffer("cheb_polynomials", torch.from_numpy(self.cheb_polynomial(L_tilde, K)))

    def scaled_Laplacian(self, W):
        '''
        Compute scaled Laplacian matrix \tilde{L} from the adjacency matrix.
        L_tilde = (2 * L) / lambda_max - I, where I is the identity matrix.
        L_tilde 特征值范围 (-1,1)

        Parameters
        ----------
        W: np.ndarray
            Adjacency matrix of shape (N, N), where N is the number of vertices.

        Returns
        -------
        scaled_Laplacian: np.ndarray
            Scaled Laplacian matrix of shape (N, N).
        '''
        assert W.shape[0] == W.shape[1]
        D = np.diag(np.sum(W, axis=1))  # 因为用的是 grid_graph, 故不会有孤立节点问题
        L = D - W
        L = L.astype(np.float32)

        # Compute the largest eigenvalue for scaling
        lambda_max = eigs(L, k=1, which='LR')[0].real
        return (2 * L) / lambda_max - np.identity(W.shape[0])

    def cheb_polynomial(self, L_tilde, K):
        '''
        Compute a list of Chebyshev polynomials from T_0 to T_{K-1}.

        Parameters
        ----------
        L_tilde: np.ndarray
            Scaled Laplacian matrix of shape (N, N).
        K: int
            The maximum order of Chebyshev polynomials.

        Returns
        -------
        cheb_polynomials: list(np.ndarray)
            List of Chebyshev polynomials from T_0 to T_{K-1}.
        '''
        N = L_tilde.shape[0]
        cheb_polynomials = [np.identity(N), np.asarray(L_tilde)]
        for i in range(2, K):
            cheb_polynomials.append(np.asarray(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]))
        cheb_polynomials = np.stack(cheb_polynomials, axis=0).astype(np.float32)
        return cheb_polynomials

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution with spatial attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, N, F_in, T).
        spatial_attention: torch.Tensor
            Spatial attention scores of shape (B, N, N), computed by spatial attention layer.

        Returns
        -------
        output: torch.Tensor
            Output tensor of shape (batch_size, N, F_out, T), where F_out is the output features.
        '''
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
                # mul是逐元素相乘
                T_k_with_at = T_k.mul(spatial_attention)
                # (in_channels, out_channels)
                theta_k = self.Theta[k]
                # (b, N, N) @ (b, N, F_in) -> (b, N, F_in)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                # (b, N, F_in) @ (F_in, F_out) -> (b, N, F_out)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))  # Add a time dimension (b, N, F_out, 1)

        # Concatenate over time and apply ReLU: (b, N, F_out, T)
        out = F.relu(torch.cat(outputs, dim=-1))
        return out



class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, in_dim * 2)
        self.conv2 = GCNConv(in_dim * 2, in_dim * 4)
        self.conv3 = GCNConv(in_dim * 4, in_dim * 2)
        self.conv4 = GCNConv(in_dim * 2, out_dim)
        self.ln = nn.LayerNorm(out_dim)

        # 线性层初始化
        # init.xavier_uniform_(self.linear.weight)
        # self.N = N

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                init.kaiming_normal_(m.lin.weight)
                # if m.bias is not None:
                    # init.zeros_(m.bias)

    def forward(self, x, adj):
        adj = adj.to(x.device)

        x = self.conv1(x, adj)  # AxW
        # x = F.relu(x)
        x = self.conv2(x, adj)
        # x = F.relu(x)
        x = self.conv3(x, adj)
        # x = F.relu(x)
        x = self.conv4(x, adj)
        # x = self.ln(x)

        return x

class GCN_spatial(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_spatial, self).__init__()
        self.conv1 = GCNConv(in_dim, in_dim * 2)
        self.conv2 = GCNConv(in_dim * 2, in_dim * 4)
        self.conv3 = GCNConv(in_dim * 4, in_dim * 2)
        self.conv4 = GCNConv(in_dim * 2,out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                init.kaiming_uniform_(m.lin.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, adj):
        adj = adj.to(x.device)
        x = x.permute(0, 2, 1) # (batch, T, Ks)->(batch, Ks, T)

        x = self.conv1(x, adj)
        # x = F.relu(x)
        x = self.conv2(x, adj)
        # x = F.relu(x)
        x = self.conv3(x, adj)
        # x = F.relu(x)
        x = self.conv4(x, adj)
        # x = self.ln(x)
        x = x.permute(0, 2, 1) # (batch, Ks, L) -> (batch, L, Ks)

        return x

class LinearPreCoeff(nn.Module):
    def __init__(self,
                 in_dim: int = 16,
                 out_dim: int = 16
                 ):
        super(LinearPreCoeff, self).__init__()

        # 定义四层全连接网络
        self.linear_pre_coeff = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),  # 第一层
            nn.ReLU(),  # 激活函数
            nn.Linear(in_dim * 2, in_dim * 4),  # 第二层
            nn.ReLU(),  # 激活函数
            nn.Linear(in_dim * 4, in_dim * 2),  # 第三层
            nn.ReLU(),  # 激活函数
            nn.Linear(in_dim * 2, out_dim)  # 第四层
        )

    def forward(self, x):
        return self.linear_pre_coeff(x)

class GATConv(nn.Module):
    # 单头的图注意力层，和GAT论文里一样
    def __init__(self, in_dim, out_dim, dropout, alpha, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):

        device = x.device  # 确保所有计算在 x.device 上

        # 增加自环
        adj = adj.clone()
        adj.fill_diagonal_(1)

        # 随机丢弃特征以防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        # 线性变换
        h = torch.matmul(x, self.weight.to(x.device))
        # h = Wx (32, 100, latent_dim=32)

        # 计算注意力分数
        N = h.size(1)  # 节点数
        batch_size = h.size(0)

        # 取边的两个端点，分别为源节点向量、目标节点向量
        # 广播计算注意力输入 [batch, N, N, 2*out_dim]
        h_i = h.unsqueeze(2).expand(batch_size, N, N, self.out_dim)  # h_i: (batch, N, N, out_dim)
        h_j = h.unsqueeze(1).expand(batch_size, N, N, self.out_dim)  # h_j: (batch, N, N, out_dim)
        a_input = torch.cat([h_i, h_j], dim=-1)  # 形状: (batch, N, N, 2*out_dim)

        # 计算注意力分数 e
        e = F.leaky_relu(torch.matmul(a_input, self.a.to(device)), negative_slope=self.alpha)  # (batch, N, N, 1)
        e = e.squeeze(-1)  # (batch, N, N)

        # 只保留邻接矩阵中的边
        e = adj * e
        # e = e.to(torch.float32) # 确保 e 是 float32
        # e = e.masked_fill(adj == 0, -1e9)

        # 归一化注意力系数
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 打印 attention 和 h 的统计信息
        # print(f"Attention matrix: min: {attention.min().item():.4f}, max: {attention.max().item():.4f}, "
        #       f"mean: {attention.mean().item():.4f}, std: {attention.std().item():.4f}")
        # print(
        #     f"h: min {h.min().item():.4f}, max {h.max().item():.4f}, "
        #     f"mean {h.mean().item():.4f}, std {h.std().item():.4f}")

        # 加权求和，计算节点的特征输出
        h_prime = torch.matmul(attention, h)
        if self.bias is not None:
            h_prime = h_prime + self.bias.to(h_prime.device)

        return h_prime


# class GraphLearningModule_NodeCentral(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout, alpha, bias=True):
#         super(GraphLearningModule_NodeCentral, self).__init__()


class GraphLearningModule_BernVAE(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, N):
        """
        Args:
            in_dim (int): 输入的原始特征维度 T
            latent_dim (int): 经过线性变换后的维度
            N (int): 节点数
        """
        super(GraphLearningModule_BernVAE, self).__init__()
        self.N = N
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, latent_dim)
        self.linear2 = nn.Linear(in_dim, latent_dim)
        self.linear_para_latent = nn.Linear(2 * latent_dim, latent_dim)
        self.linear_para_Bern = nn.Linear(latent_dim, 1)
        # self.linear_mu = nn.Linear(latent_dim, 1)
        # self.linear_logvar = nn.Linear(latent_dim, 1)

        # self.dropout = nn.Dropout(0.1)

        # 初始化参数
        self.apply(self.init_weights_bias)
        # linear1/2的初始化不同，非常有用！
        # 如果用同样的初始化，linear2的梯度会远远低于linear1，说明其功能被linear1替代
        init.xavier_normal_(self.linear2.weight, gain=1.414)

    def init_weights_bias(self, m):
        if isinstance(m, nn.Linear):
            # init.xavier_uniform_(m.weight)  # Xavier 均匀初始化
            # init.xavier_normal_(m.weight)  # Xavier 均匀初始化
            init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 适用于 ReLU
            init.zeros_(m.bias)  # bias 设为 0

    def forward(self, x):
        """
        Args:
            x: 输入数据，形状为 (B, N, T)
        返回：
            adjacency_matrix: 最终输出的稀疏邻接矩阵，形状为 (N, N)
        """
        batch_size, N, input_time_steps = x.size()
        assert N == self.N, "输入数据的节点数不匹配"

        # Step 1: 计算所有节点之间的特征差异
        # x_expanded = x.unsqueeze(2).expand(batch_size, N, N, input_time_steps)  # 形状为 (B, N, N, T)
        # diff = x_expanded - x_expanded.transpose(1, 2)  # 计算节点间的特征差异 (B, N, N, T)
        # diff = self.linear1(diff)

        # 假设节点1特征是[1,2], 节点2特征是[2,3], 节点3特征是[4,5], 即B=1,N=3,T=2，那么，
        #  x_diff shape [1,3,3,2]:
        # [[[[ 0.0, 0.0],  [ 2.0, 2.0],  [ 4.0, 4.0]],   # 节点 1 和其他节点的差异
        #   [[-2.0, -2.0],  [ 0.0, 0.0],  [ 2.0, 2.0]],   # 节点 2 和其他节点的差异
        #   [[-4.0, -4.0],  [-2.0, -2.0], [ 0.0, 0.0]]]]  # 节点 3 和其他节点的差异

        # 节点嵌入, x并列通过两个MLP
        x1 = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x2 = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        x1_expanded = x1.unsqueeze(2).expand(batch_size, N, N, self.latent_dim)  # 对 X 在维度2增加一个维度，然后扩展为 (B, N, N, T)
        x2_expanded = x2.unsqueeze(1).expand(batch_size, N, N, self.latent_dim)  # 对 Y 在维度1增加一个维度，然后扩展为 (B, N, N, T)

        # 做差值
        # diff = x1_expanded - x2_expanded # 计算x1和x2的差异, (B, N, N, T)
        # para_Bern = self.sigmoid(self.linear_para_Bern2(diff))  # (batch_size, N, N, 2T)

        # 成对串联
        x_pair = torch.cat((x1_expanded, x2_expanded), dim=-1)  # 形状为 (B, N, N, 2T)
        para_latent = F.leaky_relu(self.linear_para_latent(x_pair), negative_slope=0.2)  # 2T -> T
        # para_latent = self.dropout(para_latent)

        para_Bern = F.sigmoid(self.linear_para_Bern(para_latent))  # T -> 1
        para_Bern = para_Bern.squeeze(-1)  # [B,N,N]
        para_Bern = para_Bern.mean(dim=0)  # 对批次平均,[N, N]
        para_Bern.fill_diagonal_(0)  # 对角线置零，不然会只学到对角元

        # mu = self.linear_mu(para_latent)
        # mu = F.sigmoid(mu).squeeze(-1) + 0.5
        # mu = mu.mean(dim=0)
        # mu.fill_diagonal_(0)  # 对角线置零
        #
        # logvar = self.linear_logvar(para_latent)
        # logvar = F.sigmoid(logvar).squeeze(-1)
        # logvar = logvar.mean(dim=0)
        # logvar.fill_diagonal_(0.99)  # 对角线方差置1

        # adj_matrix_sampled = BernGauSamplesGenerate(para_Bern, mu, logvar, temperature=0.2)
        adj_matrix_sampled = BernSamplesGenerate(para_Bern, temperature=0.2, clamp_strength=10)

        # 返回最终的邻接矩阵
        return adj_matrix_sampled


class GraphLearningModule_SelfAtt_Kernel(nn.Module):
    def __init__(self, in_dim, latent_dim, N, kernel_type="rbf", initial_gamma_value=1e-3):
        """
        Args:
            in_dim (int): 输入的原始特征维度 T
            latent_dim (int): 经过线性变换后的维度
            N (int): 节点数
            kernel_type (str): 选择核函数，支持 "dot"（点积）、"cosine"（余弦相似度）、"rbf"（高斯核）、
                                         "nw"（逆距离权重）
            rbf_gamma (float): 当使用 RBF 核时的 gamma 参数
        """
        super(GraphLearningModule_SelfAtt_Kernel, self).__init__()
        self.N = N
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.kernel_type = kernel_type

        # 两个线性变换，分别作为“查询”和“键”映射
        # nn.Linear已经进行了默认的权重和偏置初始化
        self.linear1 = nn.Linear(in_dim, latent_dim)
        self.linear2 = nn.Linear(in_dim, latent_dim)

        # self.ln_dist = nn.Linear(latent_dim, 1)
        # self.sparsemax = Sparsemax(dim=1)  # 使用 sparsemax 对最终核矩阵的每行进行归一化
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.initial_gamma_value = initial_gamma_value
        self.rbf_gamma = initial_gamma_value
        # 越小越稀疏

        # 初始化参数
        self.apply(self.init_weights_bias)

        # if self.kernel_type in ["rbf"]:
        #     self.rbf_gamma = nn.Parameter(torch.tensor(initial_gamma_value,
        #                                                dtype=torch.float))
        # else:
        #     self.rbf_gamma = None

    def init_weights_bias(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform(m.weight)  # Xavier 均匀初始化
            # init.xavier_normal_(m.weight)  # Xavier 均匀初始化
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 适用于 ReLU
            init.zeros_(m.bias)  # bias 设为 0

    def kernel_function(self, X, Y):
        """
        X, Y:
            - 形状为 (B, N, dim) 的矩阵，B 为批次大小，N 为节点数，dim 为特征维度
            - 也可以是形状为 (N, dim) 的矩阵，表示单个样本的情况
        返回：形状为 (B, N, N) 的核相似度矩阵, 核值越大, 输入向量越相似
        """
        # 检查维度，确保输入 X 和 Y 是批次大小维度一致
        if X.dim() == 2:  # (N, dim)
            X = X.unsqueeze(0)  # 扩展为 (1, N, dim)
        if Y.dim() == 2:  # (N, dim)
            Y = Y.unsqueeze(0)  # 扩展为 (1, N, dim)

        B, N, dim = X.size()

        if self.kernel_type == "dot":
            # 对于每一批次，计算 X 和 Y 的点积
            return torch.bmm(X, Y.transpose(1, 2)) / (dim ** 0.5)  # (B, N, N)

        elif self.kernel_type == "cosine":
            # 归一化 X 和 Y，然后计算它们的余弦相似度
            X_norm = F.normalize(X, p=2, dim=-1)  # 归一化每个样本
            Y_norm = F.normalize(Y, p=2, dim=-1)  # 归一化每个样本
            return torch.bmm(X_norm, Y_norm.transpose(1, 2))  # (B, N, N)

        elif self.kernel_type == "rbf":
            # 计算每对节点的 RBF 核
            diff = X.unsqueeze(2) - Y.unsqueeze(1)  # (B, N, N, dim)
            dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, N, N)
            return torch.exp(-self.rbf_gamma * dist_sq)  # (B, N, N)

        elif self.kernel_type == "nw":
            # Nadaraya–Watson 核（逆距离权重）
            diff = X.unsqueeze(2) - Y.unsqueeze(1)  # (B, N, N, dim)
            dist = torch.norm(diff, dim=-1)  # (B, N, N)
            epsilon = 1e-8
            return 1.0 / (1.0 + dist + epsilon)  # (B, N, N)

        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def forward(self, x):
        """
        Args:
            x: 输入数据，形状为 (B, N, T)
        返回：
            adjacency_matrix: 最终输出的稀疏邻接矩阵，形状为 (N, N)
        """
        batch_size, N, T = x.size()
        assert N == self.N, "输入数据的节点数不匹配"

        # 对输入数据进行两个独立的线性变换，得到两个投影矩阵，形状为 (B, N, latent_dim)
        proj1 = self.linear1(x)
        proj2 = self.linear2(x)

        # 计算核函数，得到相似度矩阵，形状为 (B, N, N)
        sim_matrix = self.kernel_function(proj1, proj2)
        # sim_matrix: shape: torch.Size([8, 200, 200]), min: 0.022, max: 0.289
        # 对批次平均
        sim_matrix = sim_matrix.mean(dim=0)  # [N, N]
        sim_matrix.fill_diagonal_(0.1)
        # 将sim_matrix对角线置零，不然会只学到对角元

        # sim_matrix = self.sigmoid(sim_matrix)

        # print(f"kernel(A,B): num_edge: {torch.count_nonzero(sim_matrix).item():.2f}, "
        #       f"min: {sim_matrix.min().item():.3f}, "
        #       f"max: {sim_matrix.max().item():.3f}")

        # 使用 sparsemax 对每行进行归一化，得到稀疏的概率分布
        # sim_matrix_sparse = self.sparsemax(sim_matrix)
        # sim_matrix_sparse = self.relu(sim_matrix)
        sim_matrix_sparse = BernSamplesGenerate(sim_matrix, temperature=0.2)

        # print(f"Bern(dist): num_edge: {torch.count_nonzero(sim_matrix_sparse).item():.2f},"
        #       f"min: {sim_matrix_sparse.min().item():.3f}, "
        #       f"max: {sim_matrix_sparse.max().item():.3f}")

        # 返回最终的邻接矩阵
        return sim_matrix_sparse


class GraphLearningModule_VAE(nn.Module):
    def __init__(self, N, input_time_steps):
        super(GraphLearningModule_VAE, self).__init__()
        self.N = N  # 节点数
        self.input_time_steps = input_time_steps  # 特征维度
        self.latent_dim = input_time_steps // 2  # 潜在维度

        # 定义N个编码器，每个编码器对不同的输入进行处理
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_time_steps, self.latent_dim),
                nn.ReLU(),  # 使用 ReLU 激活函数增加稀疏性
                nn.Linear(self.latent_dim, 1),
                nn.ReLU()  # sigmoid确保输出在0到1之间, 可选
            ) for _ in range(N)])  # 对于每个节点，创建一个独立的编码器

    def forward(self, x):
        batch_size, N, input_time_steps = x.size()

        # Step 1: 计算所有节点之间的特征差异
        x_expanded = x.unsqueeze(2).expand(batch_size, N, N, input_time_steps)  # 形状为 (B, N, N, T)
        x_diff = x_expanded - x_expanded.transpose(1, 2)  # 计算节点间的特征差异 (B, N, N, T)
        # 假设节点1特征是[1,2], 节点2特征是[2,3], 节点3特征是[4,5], 即B=1,N=3,T=2，那么，
        #  x_diff shape [1,3,3,2]:
        # [[[[ 0.0, 0.0],  [ 2.0, 2.0],  [ 4.0, 4.0]],   # 节点 1 和其他节点的差异
        #   [[-2.0, -2.0],  [ 0.0, 0.0],  [ 2.0, 2.0]],   # 节点 2 和其他节点的差异
        #   [[-4.0, -4.0],  [-2.0, -2.0], [ 0.0, 0.0]]]]  # 节点 3 和其他节点的差异

        # Step 2: 将 N x T 的差异矩阵送入 N 个并行编码器
        probs = []
        for i in range(N):
            node_input = x_diff[:, :, i, :]  # 提取第i个节点的特征 (batch_size, N, input_time_steps)
            encoder_output = self.encoders[i](node_input)  # 使用第i个编码器
            prob = F.softmax(encoder_output, dim=-1)  # 在最后一个维度上归一化, 使得每个节点的输出和为 1
            probs.append(prob)  # 包含N个张量的列表, 每个张量(batch_size, N, 1)

        probs = torch.stack(probs, dim=-1).squeeze(2)  # 由列表变成tensor，形状为 (batch_size, N, N)
        adjacency_matrix = probs.mean(dim=0)  # 平均掉batch-size这个维度, (N, N)

        # print(
        #     f"Adjacency Matrix Shape: {adjacency_matrix.shape}, Total Number of Edges: {adjacency_matrix.sum().item()}")

        return adjacency_matrix


class GraphLearningModule_MLP(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, tau=1, ema_alpha=0.9):
        """
        图学习模块
        Args:
            num_nodes (int): 图中节点的数量 N。
            in_dim (int): 输入特征的维度 T。
            hidden_dim (int): MLP 的隐藏层维度。
            tau (float): Gumbel-Softmax 的温度参数。
            ema_alpha (float): EMA 平滑参数。
            sparse_lambda (float): 稀疏性正则化的权重。
        """
        super(GraphLearningModule_MLP, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.ema_alpha = ema_alpha

        # 定义 MLP 用于学习边的概率，输出是每条边的得分（logits）
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        前向传播
        Args:
            node_features (Tensor): 输入节点特征，形状为 [B, N, T]。
        Returns:
            edge_index (Tensor): 学习到的图结构，稀疏表示，形状为 [2, num_edges]。
            loss (Tensor): 当前批次的图学习损失。
        """

        B, N, T = x.size()

        # 1. 构造边特征：所有节点对的特征组合 [B, N, N, 2T]
        x_expanded1 = x.unsqueeze(2)  # (B, N, 1, T)
        x_expanded2 = x.unsqueeze(1)  # (B, 1, N, T)
        # 在对应维度上复制，使得它们在第2和第3维上大小一致
        x_expanded1 = x_expanded1.expand(-1, -1, N, -1)  # (B, N, N, T)
        x_expanded2 = x_expanded2.expand(-1, N, -1, -1)  # (B, N, N, T)

        x_pair = torch.cat((x_expanded1, x_expanded2), dim=-1)  # 形状为 (B, N, N, 2T)
        x_pair = x_pair.view(B, N * N, 2 * T)

        # 2. 学习边 logits：使用 MLP 计算每个边的未归一化分数 [Batch_size, N, N]
        # min = -1.2667, max = 1.4502, mean = -0.0863, std = 0.2264
        edge_logits_batch = self.edge_mlp(x_pair).squeeze(-1).view(B, N, N)  # [Batch_size, N, N]
        edge_logits = edge_logits_batch.mean(dim=0)  # [N, N], 对批次数求平均
        # print(f"edge_logits_batch: shape={edge_logits_batch.shape}, "
        #       f"min = {edge_logits_batch.min().item():.4f}, max = {edge_logits_batch.max().item():.4f}, "
        #       f"mean = {edge_logits_batch.mean().item():.4f}, std = {edge_logits_batch.std().item():.4f}")

        # 3. 更新共享的边概率矩阵edge_probs（EMA 平滑处理）
        # with torch.no_grad():  # 关闭梯度计算
        #     # 对批次数目求平均并 Sigmoid 归一化 [N, N]
        #     batch_mean_probs = edge_logits_batch.mean(dim=0).sigmoid()
        #     # 当前新结果与已有旧结果加权求和
        #     self.edge_probs.data = self.ema_alpha * self.edge_probs.data + (1 - self.ema_alpha) * batch_mean_probs

        # 4. 对当前的edge_logits_batch进行采样，使用Straight-Through Gumbel-Softmax
        # 把 edge_logits_batch 变成 0/1 的数值

        adj_matrix = straight_through_gumbel_softmax(logits=edge_logits,
                                                     temperature=self.tau,
                                                     hard=True)  # [N, N]
        # print(f"adj_matrix: shape={adj_matrix.shape}, "
        #       f"min = {adj_matrix.min().item():.4f}, max = {adj_matrix.max().item():.4f}, "
        #       f"mean = {adj_matrix.mean().item():.4f}, std = {adj_matrix.std().item():.4f}")

        return adj_matrix


import torch
import torch.nn as nn


class KNNGraphLearn(nn.Module):
    def __init__(self,
                 k_neighbors: int = 5,
                 return_adjacency_matrix: bool = True):

        super(KNNGraphLearn, self).__init__()
        self.k_neighbors = k_neighbors
        self.return_adjacency_matrix = return_adjacency_matrix  # 新增的参数，控制返回边索引或邻接矩阵

    def forward(self, X):
        """
        输入:
        X: 张量，形状为 (batch_size, N, T)，表示每个批次的 N 个节点和每个节点的 T 维特征。

        输出:
        edge_index 或 adjacency_matrix:
            - 如果 return_adjacency_matrix=True，返回邻接矩阵，形状为 (N, N)；
            - 如果 return_adjacency_matrix=False，返回边索引，形状为 (2, num_edges)，表示邻接矩阵的稀疏形式。
        """
        batch_size, N, T = X.shape

        # Step 1: 将不同 batch 的数据串联起来，变成 (N, T * batch_size)
        X_flat = X.permute(1, 2, 0).reshape(N, T * batch_size).cpu().detach().numpy()

        # Step 2: 使用 KNN 算法计算邻接关系
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm="auto").fit(X_flat)
        distances, indices = nbrs.kneighbors(X_flat)

        # Step 3: 生成 edge_index 和邻接矩阵
        edge_index = []
        adjacency_matrix = torch.zeros((N, N), dtype=torch.float32)  # 用于存储邻接矩阵

        for i in range(N):
            # 获取第 i 个节点的最近邻节点（排除自身）
            neighbors = indices[i][1:]  # 跳过自身节点
            for neighbor in neighbors:
                edge_index.append([i, neighbor])
                # 更新邻接矩阵
                adjacency_matrix[i, neighbor] = 1
                adjacency_matrix[neighbor, i] = 1  # 对称邻接矩阵

        # 将 edge_index 转换为 PyTorch 张量，并转置为形状 (2, num_edges)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        if self.return_adjacency_matrix:
            return adjacency_matrix
        else:
            return edge_index

class SparseCodingModule(nn.Module):
    """稀疏编码模块：匹配字典 D 获得稀疏系数 C"""

    def __init__(self, T, K):
        super(SparseCodingModule, self).__init__()
        # self.dic = nn.Parameter(torch.empty(K, T))  # 待优化字典矩阵 D
        self.dic = nn.Embedding(K, T)
        init.kaiming_normal_(self.dic.weight)  # 用xavier方法初始化

    def forward(self, X):
        # 计算稀疏系数 C = X @ D
        C = torch.matmul(X, self.dic.weight.T)
        return C, self.dic.weight

class SparseCodingSpatial(nn.Module):
    def __init__(self, Ks, N):
        super(SparseCodingSpatial, self).__init__()
        self.dic = nn.Embedding(Ks, N)  # 待优化字典矩阵 D
        init.kaiming_uniform_(self.dic.weight)  # 用xavier方法初始化
    def forward(self, X):
        X = X.permute(0, 2, 1)
        # 计算稀疏系数 C = X @ D
        C = torch.matmul(X, self.dic.weight.T)
        return C, self.dic.weight

class MultiHeadGAT(nn.Module):
    def __init__(
            self,
            in_dim=32,  # 输入特征的维度 (例如，每个节点的特征维度是 32)
            nlayers=None,  # 模型的层数（未使用）
            latent_dim=32,  # 隐藏层的输出特征维度 (每一层的GAT将特征维度变为latent_dim)
            out_dim=10,  # 输出的类别数量 (例如，用于分类的节点类别数)
            head_GAT=8,  # GAT 的多头注意力数量 (多头机制)
            nhead_out=4,  # 输出层的多头注意力数量 (最后一层的多头数量)
            alpha=0.2,  # GATconv中的LeakyReLU 的负斜率 (用于注意力机制)
            dropout=0.2,  # Dropout 概率 (防止过拟合)
            **kwargs
    ):
        super().__init__()

        self.in_dim = in_dim
        self.nlayers = nlayers
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.head_GAT = head_GAT
        self.nhead_out = nhead_out
        self.alpha = alpha
        self.dropout = dropout

        # 第一层的 GAT 多头注意力机制，单头用GATConv实现
        self.attentions = nn.ModuleList([
            GATConv(self.in_dim, self.latent_dim, dropout=self.dropout, alpha=self.alpha)
            for _ in range(self.head_GAT)
        ])
        # 在第一层输出后加一个LayerNorm，归一化拼接后的特征
        self.ln1 = nn.LayerNorm(self.latent_dim * self.head_GAT)

        # 输出层的 GAT 注意力机制 (注意，这里的多头是 nhead_out)
        self.out_atts = nn.ModuleList([
            GATConv(self.latent_dim * self.head_GAT, self.out_dim, dropout=self.dropout, alpha=self.alpha)
            for _ in range(self.nhead_out)
        ])

        self.reset_parameters()

    # 对 GATConv 中的权重、偏置进行Xavier初始化
    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self, x, adj):

        # 统一设备
        device = x.device
        edge_index = adj.to(device)

        # 1️⃣ 处理边缘 (删除自环 + 添加自环)
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2️⃣ 第一层的 GAT 计算 (多头注意力)
        # print(f"before GAT: x shape = {x.shape}")
        # before GAT: x shape = torch.Size([32, 500, 20]), (batch_size, N, K)
        x = torch.cat([att(x.to(device), edge_index) for att in self.attentions], dim=-1)
        x = F.elu(x)  # elu激活函数
        # print(f"after GAT: x shape = {x.shape}")
        # after GAT: x shape = torch.Size([32, 500, 32*3]), (batch_size, N, latent_dim * head_GAT)

        # 应用 LayerNorm 来稳定拼接后的特征
        x = self.ln1(x)

        # 3️⃣ 输出层的 GAT 计算 (多个头的输出相加)
        x = torch.stack([att(x.to(device), edge_index) for att in self.out_atts], dim=0)
        # 在头上增加一个维度，变成(nhead_out, batch_size, N, out_dim)
        x = x.mean(dim=0)  # 对头上的维度求平均，输出形状为 (batch_size, N, out_dim)

        # 打印 self.attentions 中每个 GATConv 模块的权重信息
        # for i, att in enumerate(self.attentions):
        #     out_i = att(x.to(device), edge_index)
        #     print(
        #         (f"Head {i} output: min: {out_i.min().item():.4f}, max: {out_i.max().item():.4f}, "
        #          f"mean: {out_i.mean().item():.4f}, std: {out_i.std().item():.4f}"))
        # for i, att in enumerate(self.attentions):
        #     print(f"Attention head {i} - weight: min {att.weight.min().item():.4f}, max {att.weight.max().item():.4f}, "
        #           f"mean {att.weight.mean().item():.4f}, std {att.weight.std().item():.4f}")
        #     print(f"Attention head {i} - a: min {att.a.min().item():.4f}, max {att.a.max().item():.4f}, "
        #           f"mean {att.a.mean().item():.4f}, std {att.a.std().item():.4f}")

        # 同样打印 self.out_atts 中每个模块的信息
        # for i, att in enumerate(self.out_atts):
        #     print(
        #         f"Out Attention head {i} - weight: min {att.weight.min().item():.4f}, max {att.weight.max().item():.4f}, "
        #         f"mean {att.weight.mean().item():.4f}, std {att.weight.std().item():.4f}")
        #     print(f"Out Attention head {i} - a: min {att.a.min().item():.4f}, max {att.a.max().item():.4f}, "
        #           f"mean {att.a.mean().item():.4f}, std {att.a.std().item():.4f}")
        # print(f"after GAT-out: x shape = {x.shape}")
        return x


class SigmoidDiffWindow2D(nn.Module):
    def __init__(self, T, K, L, init_a=None, init_k=None):
        """
        初始化二维 Sigmoid Diff 窗函数
        K: 窗函数的数量
        L: 窗口的长度
        init_a: 窗函数的初始起始点 (K 个值)
        init_k: 窗函数的初始陡峭度 (K 个值)
        """
        super(SigmoidDiffWindow2D, self).__init__()
        self.K = K
        self.L = L
        self.T = T

        # a 的范围是(0,1), 那么 start_idx 的范围是从 0 到 round(0.8808 * (T - self.L - 1)) 之间的整数。
        # self.a = init_a if init_a is not None else torch.ones(K) * 0.8
        # self.k = init_k if init_k is not None else torch.ones(K) * 5.0

        # 如果没有提供初始值，随机初始化
        # self.k = nn.Parameter(init_k if init_k is not None else torch.ones(K) * 5.0)  # K 个不同的陡峭度
        self.k = nn.Parameter(3 + (10 - 3) * torch.rand(K) )  # K 个不同的陡峭度

        # self.a = init_a if init_a is not None else torch.rand(K)  # K 个不同的起始点
        init_start_idx = 0.5 * (torch.tanh(torch.rand(K)) + 1) * (T - L - 1)
        self.start_idx = nn.Parameter(init_start_idx)

    def forward(self, x):
        """
        前向传播
        x: 输入数据，形状为 (K, T)
        返回：经过 Sigmoid Diff 窗函数的输出，形状为 (K, L)
        """

        # 初始化窗函数的输出矩阵

        output = torch.zeros((self.K, self.L), device=x.device)
        self.window = torch.zeros((self.K, self.T), device=x.device)

        # 对每一行应用不同的窗函数
        for i in range(self.K):
            k = self.k[i]  # 每个窗函数的陡峭度
            si = self.start_idx[i]

            # 创建 Sigmoid Diff 窗函数
            t = torch.arange(self.T, device=x.device)  # 时间轴
            # sigmoid = lambda t: 1 / (1 + torch.exp(-k * t))
            # softplus(x) = log(1 + exp(x)) 在大数值时更稳定，梯度也不会消失
            sigmoid = lambda t: 1 / (1 + torch.exp(-F.softplus(k * t)))
            epsilon = 1e-6  # 避免梯度 NaN
            window = 2 * (sigmoid(t - (si + epsilon)) - sigmoid(t - (si + self.L + epsilon)))  # Sigmoid Diff 窗函数
            weighted_row = x[i] * window  # 按窗函数加权

            si_int = SinRoundFunction.apply(si)  # 前向取整，反向近似
            self.start_idx.data[i] = si_int  # 通过创建新张量来更新而不是in-place操作
            self.window[i, :] = window

            output[i, :] = weighted_row[si_int:si_int + self.L]

        return output


class BasisExtractionModule(nn.Module):
    """基提取模块：从 D 提取时序基 D'"""

    def __init__(self, in_dim, out_dim, temporal_basis_number):
        super(BasisExtractionModule, self).__init__()

        self.sigmoid_diff = SigmoidDiffWindow2D(K=temporal_basis_number,
                                                L=out_dim, T=in_dim)  # 使用二维 Sigmoid Diff 窗函数
        # self.mlp = nn.Linear(in_dim, out_dim, bias=True) # 测试了，不如 sigmoid_diff
    def forward(self, x):
        x = self.sigmoid_diff(x)  # 应用 Sigmoid Diff 窗函数
        # x = self.mlp(x)
        return x


class PredictOnCoeff(nn.Module):
    def __init__(self, input_time_steps, coeff_number, predict_time_steps, is_print=False):
        """
        Args:
            input_time_steps: 输入的特征维度
            coeff_number: 中间层的神经元数量，也就是系数的数量
            predict_time_steps: 输出的特征维度
        """
        super(PredictOnCoeff, self).__init__()
        self.fc1 = nn.Linear(input_time_steps, coeff_number)
        self.fc2 = nn.Linear(coeff_number, coeff_number)
        self.fc3 = nn.Linear(coeff_number, predict_time_steps)
        self.relu = nn.ReLU()  # 你也可以使用其他激活函数

        # 定义正则化损失的权重（可以根据实际情况调节）
        self.lambda_fc1 = 1.0
        self.lambda_fc3 = 1.0
        self.lambda_sparse = 1.0

    def orthogonal_loss_columns(self, W):
        """
        对于权重矩阵 W (shape: [out_dim, in_dim])，
        约束其每一列互相正交。
        """
        # W^T W 的形状为 [in_dim, in_dim]
        WT_W = torch.mm(W.t(), W)
        diag = torch.diag(WT_W)
        # 构造对角矩阵
        diag_matrix = torch.diag(diag)
        # 计算非对角项的 Frobenius 范数平方
        loss = torch.norm(WT_W - diag_matrix, p='fro') ** 2
        return loss

    def orthogonal_loss_rows(self, W):
        """
        对于权重矩阵 W (shape: [out_dim, in_dim])，
        约束其每一行互相正交。
        """
        # W W^T 的形状为 [out_dim, out_dim]
        W_WT = torch.mm(W, W.t())  # 矩阵 W 和其转置 W^T 的矩阵乘积，得到一个对称矩阵 W_WT。
        diag = torch.diag(W_WT)  # 对角向量
        diag_matrix = torch.diag(diag)  # 对角矩阵
        loss = torch.norm(W_WT - diag_matrix, p='fro') ** 2
        return loss

    def forward(self, x, is_print=False):
        # x 的形状应为 (batch_size, N, input_time_steps)
        x_coeff_ori = self.fc1(x)
        x_coeff_pred = self.fc2(x_coeff_ori)
        x_pred = self.fc3(x_coeff_pred)

        # 对 fc1：要求其权重矩阵的每一列互相正交
        # fc1.weight 的 shape: [coeff_number, input_time_steps]
        loss_ortho_fc1 = self.orthogonal_loss_columns(self.fc1.weight)

        # print(f"x.shape: {x.shape}, fc1.weight.shape:{self.fc1.weight.shape}")

        # 对 fc3：要求其权重矩阵的每一行互相正交
        # fc3.weight 的 shape: [predict_time_steps, coeff_number]
        loss_ortho_fc3 = self.orthogonal_loss_rows(self.fc3.weight)

        # 总的正交损失项
        ortho_loss = self.lambda_fc1 * loss_ortho_fc1 + self.lambda_fc3 * loss_ortho_fc3

        # 计算稀疏性损失：L1范数越小越稀疏
        # 这里 lambda_sparse 是一个超参数，用于调节稀疏性损失的权重

        sparsity_loss = self.lambda_sparse * (torch.abs(x_coeff_ori).mean() + torch.abs(x_coeff_pred).mean())

        return x_pred, sparsity_loss + ortho_loss


class GLBernVAEModule(nn.Module):
    def __init__(self, N, in_dim, hidden_dim=16, latent_dim=1):
        super(GLBernVAEModule, self).__init__()

        self.node_encoder = nn.Linear(int(in_dim / 2), hidden_dim)

        # 编码器
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_para_Bern = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值 mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差 logvar
        self.bn = nn.BatchNorm2d(N)

        # 解码器
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_dim)

        self.recon_loss = nn.MSELoss()

        # 统一初始化整个模型
        self.apply(self.init_weights_bias)

    def init_weights_bias(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)  # Xavier 均匀初始化
            # init.xavier_normal_(m.weight)  # Xavier 均匀初始化
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 适用于 ReLU
            init.zeros_(m.bias)  # bias 设为 0

    def encode(self, x):
        """ 编码器：输入 x，输出 (mu, logvar) """
        h = F.relu(self.fc1(x))
        h = self.bn(h)  # 批归一化
        para_Bern = F.sigmoid(self.fc_para_Bern(h))
        mu = F.tanh(self.fc_mu(h))  # 计算均值
        logvar = F.softplus(self.fc_logvar(h))  # 限制 logvar，确保是正数
        return para_Bern, mu, logvar

    def decode(self, z):
        """ 解码器：输入潜在变量 z，输出重构结果 """
        h = F.leaky_relu(self.fc2(z))
        y = self.fc3(h)
        return y

    def forward(self, x):
        batch_size, N, T = x.shape

        # node encoder, [B,N,T]->[B,N,hidden_dim]
        x = F.leaky_relu(self.node_encoder(x))

        x1_expanded = x.unsqueeze(2).expand(batch_size, N, N, -1)  # 对 X 在维度2增加一个维度，然后扩展为 (B, N, N, T)
        x2_expanded = x.unsqueeze(1).expand(batch_size, N, N, -1)  # 对 Y 在维度1增加一个维度，然后扩展为 (B, N, N, T)
        x_pair = torch.cat((x1_expanded, x2_expanded), dim=-1)  # 形状为 (B, N, N, 2T)

        # """ 前向传播：x -> (mu, logvar) -> z -> 重构 x' """
        para_Bern, mu, logvar = self.encode(x_pair)  # 编码
        # para_Bern.diagonal(dim1=-2, dim2=-1).zero_() # 对角线置零，不然会只学到对角元
        # z = BernSamplesGenerate(para_Bern.squeeze(-1), temperature = 0.2) * GauSamplesGenerate(mu.squeeze(-1), logvar.squeeze(-1))  # [B, N, N]
        # z = GauSamplesGenerate(mu.squeeze(-1), logvar.squeeze(-1))  # [B, N, N]
        z = BernSamplesGenerate(para_Bern.squeeze(-1), temperature=0.2)  # [B, N, N]

        recon_x = self.decode(z.unsqueeze(-1))  # 生成重构 x'

        adj = z.mean(dim=0)
        loss_recon = self.recon_loss(recon_x, x_pair)

        return adj, loss_recon  # 返回重构结果和变分参数


class MySingleWAT(STBase):
    """仅有时域分解"""

    def __init__(self, N, input_time_steps, K, L,
                 Ks = 18,
                 *args, **kwargs):
        super(MySingleWAT, self).__init__(*args, **kwargs)
        # self.graph_learning = GraphLearningModule_BernVAE(in_dim=input_time_steps, latent_dim=16, out_dim=1, N=N)
        # self.graph_learning = KNNGraphLearn(k_neighbors=3, return_adjacency_matrix=True)
        self.sparse_coding = SparseCodingModule(T=input_time_steps, K=K)
        self.graphNN = GCN(in_dim=K, out_dim=K)
        # self.graphNN = MultiHeadGAT(head_GAT=4, nhead_out=1,in_dim=K,out_dim=K, N=N)
        # self.linear_pre_coeff = LinearPreCoeff(in_dim=K, out_dim=K)
        self.basis_extraction = BasisExtractionModule(in_dim=input_time_steps, out_dim=L, temporal_basis_number=K)
        self.basisconv = nn.Conv1d(in_channels=K, out_channels=K, kernel_size=3, padding=0, dilation=1, stride=4)
        self.timeconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding=(0, 1))
        self.ln = nn.LayerNorm([N, L])
        self.linear_layer = nn.Linear(N, N)
        self.finalconv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

        self.seq_len = input_time_steps
        self.N = N
        self.sqrtN = int(N ** 0.5)
        self.K = K
        self.pred_len = L
        self.lambda_reconstruct = 0.02
        self.lambda_graph = 1 / (N ** 1)
        self.lambda_graph_KL = 1
        self.lambda_sparse = 1 / (N * K)
        self.lambda_orthogonality = 0.01

    def _compute_forward_results(self, X):
        """ 计算模型的所有中间变量，避免 `forward` 和 `log_intermediate_results` 代码重复 """

        # X: [batch_size, N, T]
        # X = self.timeconv(X.unsqueeze(1)).squeeze(1) # => [batch_size, N, T]

        # 1. 图学习模块
        # A_unnorm = self.graph_learning(X) # [N, N]
        A_unnorm = np.load("/data/scratch/jiayin/Adj_SmoothGL_Milan10Min_Internet.npy")
        # A_unnorm = torch.eye(self.N, device = X.device)
        # A_unnorm = nx.adjacency_matrix(nx.grid_2d_graph(self.sqrtN, self.sqrtN)) # grid graph
        # A_unnorm = nx.adjacency_matrix(nx.erdos_renyi_graph(self.N, 3.8/self.N, seed=42))

        A = Adj2EdgeList(A_unnorm, filter_small_values = True, print_warning = False) # edge index
        # A = normalize_adj_add_self_hoop(A_unnorm)
        # A = torch.ones(2, self.N, device = X.device).int()

        # 2. 稀疏编码模块
        C, D = self.sparse_coding(X) # C: [batch_size, N, K], D: [K, T]
        # 3. GCN 模块
        C_pre = self.graphNN(x=C, adj=A) # => [batch_size, N, K]
        # 4. 基提取模块
        D_ext = self.basis_extraction(D) # => [K, L]
        # 5. 预测模块
        X_pre_temporal = torch.matmul(C_pre, D_ext) # => [batch_size, N, L]

        X_pre = X_pre_temporal

        # 6. 计算各项损失
        reconstruct_loss = self.lambda_reconstruct * nn.MSELoss()(torch.matmul(C, D), X)

        loss_graph = self.lambda_graph_KL * BernKLDiv(A, prior=0.2)
        sparsity_loss = (self.lambda_sparse * (torch.abs(C).sum()) + self.lambda_sparse * (torch.abs(C_pre).sum()))

        D_gram_matrix = torch.matmul(D, D.T)
        I_Dt = torch.eye(D.shape[0], device=D_gram_matrix.device)
        D_orthogonality_loss = self.lambda_orthogonality * torch.norm(D_gram_matrix - I_Dt, p = 'fro')

        return {
            "X_pre": X_pre,
            "A_unnorm": A_unnorm,
            "A": A,
            "C": C,
            "D": D,
            "C_pre": C_pre,
            "D_ext": D_ext,
            "reconstruct_loss": reconstruct_loss,
            "loss_graph": loss_graph,
            "sparsity_loss": sparsity_loss,
            "D_orthogonality_loss": D_orthogonality_loss,
        }

    def forward(self, X):
        """前向传播调用 `_compute_forward_results`"""
        results = self._compute_forward_results(X)
        return results["X_pre"], results["reconstruct_loss"] + results["D_orthogonality_loss"]

    def _log_intermediate_results(self, X):
        """记录 WAT 模型的中间结果到 WandB"""
        X = X.to(self.device)
        wandb_logger = self.logger.experiment  # ✅ 使用 Lightning 传入的 wandb
        results = self._compute_forward_results(X)

        A, C, D, C_pre, D_ext, X_pre = results["A"], results["C"], results["D"], results["C_pre"], results["D_ext"], \
        results["X_pre"]

        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # ✅ 先转换成 NumPy，并进行 reshape
        C_reshaped = C.detach().cpu().numpy().reshape(C.shape[0] * C.shape[1], C.shape[2])
        C_pre_reshaped = C_pre.detach().cpu().numpy().reshape(C_pre.shape[0] * C_pre.shape[1], C_pre.shape[2])
        X_pre_reshaped = X_pre.detach().cpu().numpy().reshape(X_pre.shape[0] * X_pre.shape[1], X_pre.shape[2])

        # ✅ 将矩阵转换为 Pandas DataFrame 并存储为 CSV
        pd.DataFrame(A.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "A_matrix.csv"), index=False)
        pd.DataFrame(C_reshaped).to_csv(os.path.join(save_dir, "C_matrix.csv"), index=False)
        pd.DataFrame(D.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "D_matrix.csv"), index=False)
        pd.DataFrame(C_pre_reshaped).to_csv(os.path.join(save_dir, "C_pre_matrix.csv"), index=False)
        pd.DataFrame(D_ext.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "D_ext_matrix.csv"), index=False)
        pd.DataFrame(X_pre_reshaped).to_csv(os.path.join(save_dir, "X_pre_matrix.csv"), index=False)

        # 画 A（邻接矩阵）
        fig_A, ax_A = plt.subplots(figsize=(6, 6))
        im_A = ax_A.imshow(A.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        ax_A.set_title("Adjacency Matrix (A)")
        ax_A.set_xlabel("Nodes")
        ax_A.set_ylabel("Nodes")
        plt.colorbar(im_A, ax=ax_A)

        # 存储至本地
        a_ext_img_path = os.path.join(save_dir, "Adjacency.png")
        fig_A.savefig(a_ext_img_path, format='png')
        # 存储至wandb
        img_buf_A = io.BytesIO()
        plt.savefig(img_buf_A, format='png')
        img_buf_A.seek(0)
        img_A_pil = Image.open(img_buf_A)  # ✅ 这一步转换
        plt.close(fig_A)

        # ✅ 生成第一张图（D, D_ext 的 1,5,10行）
        fig_D, axes_D = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))

        rows_to_plot = [0, int(self.K/2), self.K-1]  # 需要绘制的行索引
        for i, row in enumerate(rows_to_plot):
            axes_D[i, 0].plot(D[row, :].cpu().detach().numpy(), label=f'D Row {row}')
            axes_D[i, 0].grid(True)
            axes_D[i, 0].set_title(f'D - Row {row}')
            axes_D[i, 0].legend()

            # 计算 FFT 变换
            time_series = D[row, :].cpu().detach().numpy()
            fft_values = np.fft.fft(time_series)  # 计算 FFT
            fft_magnitudes = np.abs(fft_values)  # 计算振幅
            fft_frequencies = np.fft.fftfreq(len(time_series))  # 计算频率轴
            # 画频域信号（右列）
            axes_D[i, 1].plot(fft_frequencies[:len(fft_frequencies) // 2], fft_magnitudes[:len(fft_magnitudes) // 2],
                              label=f'FFT Row {row}')
            axes_D[i, 1].grid(True)
            axes_D[i, 1].set_title(f'FFT - Row {row}')
            axes_D[i, 1].legend()

            axes_D[i, 2].plot(D_ext[row, :].cpu().detach().numpy(), label=f'D_ext Row {row}')
            axes_D[i, 2].grid(True)
            axes_D[i, 2].set_title(f'D_ext - Row {row}')
            axes_D[i, 2].legend()

        plt.tight_layout()

        # 存储至本地
        d_ext_img_path = os.path.join(save_dir, "D_Dext_plot.png")
        fig_D.savefig(d_ext_img_path, format='png')
        # 存储至wandb
        img_buf_D = io.BytesIO()
        plt.savefig(img_buf_D, format='png')
        img_buf_D.seek(0)
        img_D_pil = Image.open(img_buf_D)  # 转换为 PIL 图像
        plt.close(fig_D)

        # 生成第二张图（C, C_pre 的 [0, :, 1], [0, :, 5], [0, :, 10]）
        fig_C, axes_C = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))

        cols_to_plot = [0, int(self.K/2), self.K-1] # [b, N, K]
        for i, col in enumerate(cols_to_plot):
            axes_C[i, 0].plot(C[0, :, col].cpu().detach().numpy(), label=f'C [0, :, {col}]')
            axes_C[i, 0].grid(True)
            axes_C[i, 0].set_title(f'C - [0, :, {col}]')
            axes_C[i, 0].legend()

            axes_C[i, 1].plot(C_pre[0, :, col].cpu().detach().numpy(), label=f'C_pre [0, :, {col}]')
            axes_C[i, 1].grid(True)
            axes_C[i, 1].set_title(f'C_pre - [0, :, {col}]')
            axes_C[i, 1].legend()

        plt.tight_layout()

        # 存储至本地
        c_ext_img_path = os.path.join(save_dir, "C_Cpre_plot.png")
        fig_C.savefig(c_ext_img_path, format='png')
        # 存储至wandb
        img_buf_C = io.BytesIO()
        plt.savefig(img_buf_C, format='png')
        img_buf_C.seek(0)
        img_C_pil = Image.open(img_buf_C)  # 转换为 PIL 图像
        plt.close(fig_C)

        wandb_logger.log({
            "ImResult/WAT_D_D_ext": wandb.Image(img_D_pil),
            "ImResult/WAT_C_C_pre": wandb.Image(img_C_pil),
            "ImResult/WAT_Adjacency_Matrix (A)": wandb.Image(img_A_pil),
        })

        # ✅ 记录数值信息到 `wandb`
        imresult_table = wandb.Table(columns=["Metric", "Value"],
                                     data=[
            ["Input_X_min", float(X.min().item())],
            ["Input_X_max", float(X.max().item())],
            ["Final_output_X_pre_min", float(X_pre.min().item())],
            ["Final_output_X_pre_max", float(X_pre.max().item())],
            ["GraphLearning_num_edge", int(torch.count_nonzero(A).item())],
            ["GraphLearning_SumA", float(A.sum().item())],
            ["SparseCoding_C_min", float(C.min().item())],
            ["SparseCoding_D_min", float(D.min().item())],
            ["Loss_Graph_Loss", float(results["loss_graph"].item())],
            ["Loss_Sparsity_Loss", float(results["sparsity_loss"].item())],
            ["Loss_D_Orthogonality_Loss", float(results["D_orthogonality_loss"].item())],
            ["Loss_Reconstruct_Loss", float(results["reconstruct_loss"].item())],
        ])
        wandb_logger.log({"ImResult_Table": imresult_table})


class MyDualWAT(STBase):
    """spatial + temporal """
    # 相比于 singleWAT，MAE和RMSE稍微低一点点，但MAPE高

    def __init__(self, N, input_time_steps, K, L,
                 Ks = 18,
                 *args, **kwargs):
        super(MyDualWAT, self).__init__(*args, **kwargs)
        self.graph_learning = GraphLearningModule_BernVAE(in_dim=input_time_steps, latent_dim=16, out_dim=1, N=N)
        # self.graph_learning = KNNGraphLearn(k_neighbors=3, return_adjacency_matrix=True)
        self.sparse_coding = SparseCodingModule(T=input_time_steps, K=K)
        self.sparse_coding_spatial = SparseCodingSpatial(N=N, Ks=Ks)
        self.graphNN = GCN(in_dim=K, out_dim=K)
        self.graphNN_spatial = GCN_spatial(in_dim=input_time_steps, out_dim=L)
        # self.graphNN = MultiHeadGAT(head_GAT=4, nhead_out=1,in_dim=K,out_dim=K, N=N)
        # self.linear_pre_coeff = LinearPreCoeff(in_dim=K, out_dim=K)
        self.basis_extraction = BasisExtractionModule(in_dim=input_time_steps, out_dim=L, temporal_basis_number=K)
        self.basisconv = nn.Conv1d(in_channels=K, out_channels=K, kernel_size=3, padding=0, dilation=1, stride=4)
        self.timeconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding=(0, 1))
        self.ln = nn.LayerNorm([N, L])
        self.linear_layer = nn.Linear(N, N)
        self.finalconv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

        self.seq_len = input_time_steps
        self.N = N
        self.sqrtN = int(N**0.5)
        self.K = K
        self.pred_len = L
        self.lambda_reconstruct = 0.02
        self.lambda_graph = 1 / (N ** 1)
        self.lambda_graph_KL = 1
        self.lambda_sparse = 1 / (N * K)
        self.lambda_orthogonality = 0.01

    def _compute_forward_results(self, X):
        """ 计算模型的所有中间变量，避免 `forward` 和 `log_intermediate_results` 代码重复 """

        # X: [batch_size, N, T]
        # X = self.timeconv(X.unsqueeze(1)).squeeze(1) # => [batch_size, N, T]

        # 1. 图学习模块
        A_unnorm = self.graph_learning(X) # [N, N]
        # A_unnorm = torch.eye(self.N, device = X.device)
        # A_unnorm = nx.adjacency_matrix(nx.erdos_renyi_graph(self.N, 3.8/self.N, seed=42))
        A = Adj2EdgeList(A_unnorm) # edge index
        # A = normalize_adj_add_self_hoop(A_unnorm)
        # A = torch.ones(2, self.N, device = X.device).int()

        # 2. 稀疏编码模块
        C, D = self.sparse_coding(X) # C: [batch_size, N, K], D: [K, T]
        # 3. GCN 模块
        C_pre = self.graphNN(x=C, adj=A) # => [batch_size, N, K]
        # 4. 基提取模块
        D_ext = self.basis_extraction(D) # => [K, L]
        # 5. 预测模块
        X_pre_temporal = torch.matmul(C_pre, D_ext) # => [batch_size, N, L]

        # 空间维度对称再来一遍
        As = torch.ones(2, self.seq_len, device = X.device).int()
        Cs, Ds = self.sparse_coding_spatial(X) # Cs: [batch_size, T, Ks], Ds: [Ks, N]
        Cs_pre = self.graphNN_spatial(x=Cs, adj=As) # => [batch_size, L, Ks]
        Ds_ext = self.linear_layer(Ds) # => [Ks, N]
        X_pre_spatial = torch.matmul(Cs_pre, Ds_ext) # => [batch_size, L, N]
        X_pre_spatial = X_pre_spatial.permute(0, 2, 1) # => [batch_size, N, L]

        X_pre = 0.5 * X_pre_spatial + 0.5 * X_pre_temporal

        # 6. 计算各项损失
        reconstruct_loss = self.lambda_reconstruct * nn.MSELoss()(torch.matmul(C, D), X)
        reconstruct_loss += self.lambda_reconstruct * nn.MSELoss()(torch.matmul(Cs, Ds), X.permute(0, 2, 1))

        loss_graph = self.lambda_graph_KL * BernKLDiv(A, prior=0.2)
        sparsity_loss = (self.lambda_sparse * (torch.abs(C).sum()) + self.lambda_sparse * (torch.abs(C_pre).sum()))

        D_gram_matrix = torch.matmul(D, D.T)
        D_gram_spatial = torch.matmul(Ds, Ds.T)
        I_Dt = torch.eye(D.shape[0], device=D_gram_matrix.device)
        I_Ds = torch.eye(Ds.shape[0], device=D_gram_spatial.device)
        D_orthogonality_loss = self.lambda_orthogonality * torch.norm(D_gram_matrix - I_Dt, p = 'fro')
        D_orthogonality_loss += self.lambda_orthogonality * torch.norm(D_gram_spatial - I_Ds, p = 'fro')

        return {
            "X_pre": X_pre,
            "A_unnorm": A_unnorm,
            "A": A,
            "C": C,
            "D": D,
            "C_pre": C_pre,
            "D_ext": D_ext,
            "reconstruct_loss": reconstruct_loss,
            "loss_graph": loss_graph,
            "sparsity_loss": sparsity_loss,
            "D_orthogonality_loss": D_orthogonality_loss,
        }

    def forward(self, X):
        """前向传播调用 `_compute_forward_results`"""
        results = self._compute_forward_results(X)
        return results["X_pre"], results["reconstruct_loss"] + results["D_orthogonality_loss"]

    def _log_intermediate_results(self, X):
        """记录 WAT 模型的中间结果到 WandB"""
        X = X.to(self.device)
        wandb_logger = self.logger.experiment  # ✅ 使用 Lightning 传入的 wandb
        results = self._compute_forward_results(X)

        A, C, D, C_pre, D_ext, X_pre = results["A"], results["C"], results["D"], results["C_pre"], results["D_ext"], \
        results["X_pre"]

        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # ✅ 先转换成 NumPy，并进行 reshape
        C_reshaped = C.detach().cpu().numpy().reshape(C.shape[0] * C.shape[1], C.shape[2])
        C_pre_reshaped = C_pre.detach().cpu().numpy().reshape(C_pre.shape[0] * C_pre.shape[1], C_pre.shape[2])
        X_pre_reshaped = X_pre.detach().cpu().numpy().reshape(X_pre.shape[0] * X_pre.shape[1], X_pre.shape[2])

        # ✅ 将矩阵转换为 Pandas DataFrame 并存储为 CSV
        pd.DataFrame(A.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "A_matrix.csv"), index=False)
        pd.DataFrame(C_reshaped).to_csv(os.path.join(save_dir, "C_matrix.csv"), index=False)
        pd.DataFrame(D.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "D_matrix.csv"), index=False)
        pd.DataFrame(C_pre_reshaped).to_csv(os.path.join(save_dir, "C_pre_matrix.csv"), index=False)
        pd.DataFrame(D_ext.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "D_ext_matrix.csv"), index=False)
        pd.DataFrame(X_pre_reshaped).to_csv(os.path.join(save_dir, "X_pre_matrix.csv"), index=False)

        # 画 A（邻接矩阵）
        fig_A, ax_A = plt.subplots(figsize=(6, 6))
        im_A = ax_A.imshow(A.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        ax_A.set_title("Adjacency Matrix (A)")
        ax_A.set_xlabel("Nodes")
        ax_A.set_ylabel("Nodes")
        plt.colorbar(im_A, ax=ax_A)

        # 存储至本地
        a_ext_img_path = os.path.join(save_dir, "Adjacency.png")
        fig_A.savefig(a_ext_img_path, format='png')
        # 存储至wandb
        img_buf_A = io.BytesIO()
        plt.savefig(img_buf_A, format='png')
        img_buf_A.seek(0)
        img_A_pil = Image.open(img_buf_A)  # ✅ 这一步转换
        plt.close(fig_A)

        # ✅ 生成第一张图（D, D_ext 的 1,5,10行）
        fig_D, axes_D = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))

        rows_to_plot = [0, int(self.K/2), self.K-1]  # 需要绘制的行索引
        for i, row in enumerate(rows_to_plot):
            axes_D[i, 0].plot(D[row, :].cpu().detach().numpy(), label=f'D Row {row}')
            axes_D[i, 0].grid(True)
            axes_D[i, 0].set_title(f'D - Row {row}')
            axes_D[i, 0].legend()

            # 计算 FFT 变换
            time_series = D[row, :].cpu().detach().numpy()
            fft_values = np.fft.fft(time_series)  # 计算 FFT
            fft_magnitudes = np.abs(fft_values)  # 计算振幅
            fft_frequencies = np.fft.fftfreq(len(time_series))  # 计算频率轴
            # 画频域信号（右列）
            axes_D[i, 1].plot(fft_frequencies[:len(fft_frequencies) // 2], fft_magnitudes[:len(fft_magnitudes) // 2],
                              label=f'FFT Row {row}')
            axes_D[i, 1].grid(True)
            axes_D[i, 1].set_title(f'FFT - Row {row}')
            axes_D[i, 1].legend()

            axes_D[i, 2].plot(D_ext[row, :].cpu().detach().numpy(), label=f'D_ext Row {row}')
            axes_D[i, 2].grid(True)
            axes_D[i, 2].set_title(f'D_ext - Row {row}')
            axes_D[i, 2].legend()

        plt.tight_layout()

        # 存储至本地
        d_ext_img_path = os.path.join(save_dir, "D_Dext_plot.png")
        fig_D.savefig(d_ext_img_path, format='png')
        # 存储至wandb
        img_buf_D = io.BytesIO()
        plt.savefig(img_buf_D, format='png')
        img_buf_D.seek(0)
        img_D_pil = Image.open(img_buf_D)  # 转换为 PIL 图像
        plt.close(fig_D)

        # 生成第二张图（C, C_pre 的 [0, :, 1], [0, :, 5], [0, :, 10]）
        fig_C, axes_C = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))

        cols_to_plot = [0, int(self.K/2), self.K-1]
        for i, col in enumerate(cols_to_plot):
            axes_C[i, 0].plot(C[0, :, col].cpu().detach().numpy(), label=f'C [0, :, {col}]')
            axes_C[i, 0].grid(True)
            axes_C[i, 0].set_title(f'C - [0, :, {col}]')
            axes_C[i, 0].legend()

            axes_C[i, 1].plot(C_pre[0, :, col].cpu().detach().numpy(), label=f'C_pre [0, :, {col}]')
            axes_C[i, 1].grid(True)
            axes_C[i, 1].set_title(f'C_pre - [0, :, {col}]')
            axes_C[i, 1].legend()

        plt.tight_layout()

        # 存储至本地
        c_ext_img_path = os.path.join(save_dir, "C_Cpre_plot.png")
        fig_C.savefig(c_ext_img_path, format='png')
        # 存储至wandb
        img_buf_C = io.BytesIO()
        plt.savefig(img_buf_C, format='png')
        img_buf_C.seek(0)
        img_C_pil = Image.open(img_buf_C)  # 转换为 PIL 图像
        plt.close(fig_C)

        wandb_logger.log({
            "ImResult/WAT_D_D_ext": wandb.Image(img_D_pil),
            "ImResult/WAT_C_C_pre": wandb.Image(img_C_pil),
            "ImResult/WAT_Adjacency_Matrix (A)": wandb.Image(img_A_pil),
        })

        # ✅ 记录数值信息到 `wandb`
        imresult_table = wandb.Table(columns=["Metric", "Value"],
                                     data=[
            ["Input_X_min", float(X.min().item())],
            ["Input_X_max", float(X.max().item())],
            ["Final_output_X_pre_min", float(X_pre.min().item())],
            ["Final_output_X_pre_max", float(X_pre.max().item())],
            ["GraphLearning_num_edge", int(torch.count_nonzero(A).item())],
            ["GraphLearning_SumA", float(A.sum().item())],
            ["SparseCoding_C_min", float(C.min().item())],
            ["SparseCoding_D_min", float(D.min().item())],
            ["Loss_Graph_Loss", float(results["loss_graph"].item())],
            ["Loss_Sparsity_Loss", float(results["sparsity_loss"].item())],
            ["Loss_D_Orthogonality_Loss", float(results["D_orthogonality_loss"].item())],
            ["Loss_Reconstruct_Loss", float(results["reconstruct_loss"].item())],
        ])
        wandb_logger.log({"ImResult_Table": imresult_table})