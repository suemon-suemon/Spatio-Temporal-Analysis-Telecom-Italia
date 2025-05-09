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
from utils.registry import register

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

@register("DualWAT")
class MyDualWAT(STBase):
    """spatial + temporal """
    # 相比于 singleWAT，MAE和RMSE稍微低一点点，但MAPE高

    def __init__(self,
                 close_len,
                 pred_len,
                 time_basis_number,
                 spatial_basis_number,
                 n_col, n_row,
                 lambda_reconstruct: float = 0.02,
                 lambda_orthogonality: float = 0.01,
                 *args, **kwargs):
        super(MyDualWAT, self).__init__(*args, **kwargs)

        self.N = n_col * n_row
        self.n_row = n_row
        self.n_col = n_col
        self.K = time_basis_number
        self.pred_len = pred_len
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_orthogonality = lambda_orthogonality

        # self.graph_learning = GraphLearningModule_BernVAE(in_dim=input_time_steps, latent_dim=16, out_dim=1, N=N)
        # self.graph_learning = KNNGraphLearn(k_neighbors=3, return_adjacency_matrix=True)
        self.sparse_coding = SparseCodingModule(T=close_len, K=time_basis_number)
        self.sparse_coding_spatial = SparseCodingSpatial(N=self.N, Ks=spatial_basis_number)
        self.graphNN = GCN(in_dim=time_basis_number, out_dim=time_basis_number)
        self.graphNN_spatial = GCN_spatial(in_dim=close_len, out_dim=pred_len)
        # self.graphNN = MultiHeadGAT(head_GAT=4, nhead_out=1,in_dim=K,out_dim=K, N=N)
        # self.linear_pre_coeff = LinearPreCoeff(in_dim=K, out_dim=K)
        self.basis_extraction = BasisExtractionModule(in_dim=close_len, out_dim=pred_len, temporal_basis_number=time_basis_number)
        self.basisconv = nn.Conv1d(in_channels=time_basis_number, out_channels=time_basis_number, kernel_size=3, padding=0, dilation=1, stride=4)
        self.timeconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding=(0, 1))
        self.ln = nn.LayerNorm([self.N, pred_len])
        self.linear_layer = nn.Linear(self.N, self.N)
        self.finalconv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

        # 目前没用到这几个正则化参数
        self.lambda_graph = 1 / (self.N ** 1)
        self.lambda_graph_KL = 1
        self.lambda_sparse = 1 / (self.N * time_basis_number)

        # 配合 STbase
        self.seq_len = close_len

    def _compute_forward_results(self, X):
        """ 计算模型的所有中间变量，避免 `forward` 和 `log_intermediate_results` 代码重复 """

        # X: [batch_size, N, T]
        # X = self.timeconv(X.unsqueeze(1)).squeeze(1) # => [batch_size, N, T]

        # 1. 图学习模块
        # A_unnorm = self.graph_learning(X) # [N, N]
        # A_unnorm = torch.eye(self.N, device = X.device)
        A_unnorm = nx.adjacency_matrix(nx.grid_2d_graph(self.n_row, self.n_col))
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
        As = torch.ones(2, self.seq_len, device = X.device).int() # 时间图
        # As_unnorm = nx.adjacency_matrix(nx.grid_2d_graph(self.N, self.N))
        # As = Adj2EdgeList(As_unnorm)
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