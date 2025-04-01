# -*- coding:utf-8 -*-
# code from https://github.com/Davidham3/ASTGCN
#           https://github.com/guoshnBJTU/ASTGCN-r-pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
from einops import rearrange, repeat
from models.STBase import STBase


class Spatial_Attention_layer(nn.Module):
    '''
    Compute spatial attention scores for the graph based on input features.

    Parameters
    ----------
    in_channels: int
        The number of input channels (features per vertex).
    num_of_vertices: int
        The number of vertices (nodes) in the graph.
    num_of_timesteps: int
        The number of timesteps (time duration of input data).

    Attributes
    ----------
    W1, W2, W3: torch.nn.Parameter
        Learnable parameters for spatial attention layer.
    bs: torch.nn.Parameter
        Bias for spatial attention computation.
    Vs: torch.nn.Parameter
        Learnable weight matrix for final spatial attention score computation.
    '''

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))

    def forward(self, x):
        '''
        Forward pass for spatial attention computation.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, N, F_in, T), where:
            - batch_size: number of batches
            - N: number of vertices (nodes)
            - F_in: number of input features
            - T: number of timesteps

        Returns
        -------
        S_normalized: torch.Tensor
            Normalized spatial attention scores of shape (B, N, N), where:
            - B: batch size
            - N: number of vertices
        '''
        # 和 temporal attention 原理基本一致，只是操作的维度不一样
        # temporal attention 是把空间维度平均掉，而 spatial attention 是把时间维度平均掉
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b, N, F, T) -> (b, N, F) -> (b, N, T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F) -> (b, N, T)
        product = torch.matmul(lhs, rhs)  # (b, N, T) @ (b, T, N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N, N) @ (B, N, N) -> (B, N, N)
        S_normalized = F.softmax(S, dim=1)  # Apply softmax to normalize the attention scores
        return S_normalized


class cheb_conv_withSAt(nn.Module):
    #     '''
    #     K-order Chebyshev graph convolution with spatial attention.
    #
    #     Parameters
    #     ----------
    #     K: int
    #         The order of the Chebyshev polynomial.
    #     in_channels: int
    #         The number of input channels (features per vertex).
    #     out_channels: int
    #         The number of output channels (features per vertex).
    #     L_tilde: torch.Tensor
    #         The scaled Laplacian matrix of the graph.
    #
    #     Attributes
    #     ----------
    #     Theta: torch.nn.ParameterList
    #         List of learnable parameters for each Chebyshev filter.
    #     cheb_polynomials: torch.Tensor
    #         Precomputed Chebyshev polynomials of order K.
    #     '''

    def __init__(self, K, in_channels, out_channels, L_tilde):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.in_channels = in_channels # 固定是1，输入的每个节点的特征维度
        self.out_channels = out_channels # nb_chev_filter, 默认是64，输出的每个节点的特征维度
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.register_buffer("cheb_polynomials", torch.from_numpy(cheb_polynomial(L_tilde, K)))

    # def forward(self, x, spatial_attention):
    #     batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
    #     outputs = []
    #     for time_step in range(num_of_timesteps): # 对每个时间步单独算卷积，怪不得慢
    #         graph_signal = x[:, :, :, time_step]  # Extract features for the current time step (b, N, F_in)
    #         output = torch.zeros(batch_size, num_of_vertices, self.out_channels).type_as(x)  # (b, N, F_out)
    #         for k in range(self.K): # 对每一阶chev多项式
    #             # chev 多项式 Tk (N, N), 由 L_tilde 矩阵计算得到
    #             # L_tilde ：网格图结构 + 特征值归一化
    #             T_k = self.cheb_polynomials[k]
    #             # chev 多项式 Tk (N, N) * spatial_attention (b, N, N) -> (b, N, N)
    #             # mul是逐元素相乘
    #             T_k_with_at = T_k.mul(spatial_attention)
    #             # (in_channels, out_channels)
    #             theta_k = self.Theta[k]
    #             # (b, N, N) @ (b, N, F_in) -> (b, N, F_in)
    #             rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
    #             # (b, N, F_in) @ (F_in, F_out) -> (b, N, F_out)
    #             output = output + rhs.matmul(theta_k)
    #         outputs.append(output.unsqueeze(-1))  # Add a time dimension (b, N, F_out, 1)
    #
    #     # Concatenate over time and apply ReLU: (b, N, F_out, T)
    #     out = F.relu(torch.cat(outputs, dim=-1))
    #     return out

    def forward(self, x, spatial_attention):

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        # Reshape the input tensor to combine batch_size and time steps
        x_reshaped = x.view(batch_size * num_of_timesteps, num_of_vertices, in_channels)  # (batch_size * T, N, F_in)

        # Prepare for accumulating results
        output = torch.zeros(batch_size * num_of_timesteps, num_of_vertices, self.out_channels).type_as(x)  # (batch_size * T, N, F_out)

        for k in range(self.K):  # For each Chebyshev polynomial
            # Get the Chebyshev polynomial T_k (b, N, N)
            T_k = self.cheb_polynomials[k]

            # Apply spatial attention element-wise multiplication (b, N, N) * (b, N, N)
            T_k_with_at = T_k.mul(spatial_attention)  # (b, N, N)

            # Expand T_k_with_at to match the shape of x_reshaped (b, T, N, N) @ (b * T, N, F_in)
            T_k_with_at_expanded = T_k_with_at.unsqueeze(1).expand(batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)  # (b, T, N, N)

            # Reshape T_k_with_at to (b * T, N, N)
            T_k_with_at_expanded = T_k_with_at_expanded.contiguous().view(batch_size * num_of_timesteps, num_of_vertices, num_of_vertices)

            # Compute the graph signal multiplication: (b * T, N, N) @ (b * T, N, F_in) -> (b * T, N, F_in)
            rhs = T_k_with_at_expanded.permute(0, 2, 1).matmul(x_reshaped)  # (b * T, N, F_in)

            # Apply the linear transformation (b * T, N, F_in) @ (F_in, F_out) -> (b * T, N, F_out)
            output = output + rhs.matmul(self.Theta[k])  # Accumulate results

        # Reshape the output back to the original batch size and time steps
        output = output.view(batch_size, num_of_timesteps, num_of_vertices, self.out_channels)  # (batch_size, T, N, F_out)

        # Apply ReLU and transpose to get the final output
        out = F.relu(output.permute(0, 2, 3, 1))  # (batch_size, N, F_out, T)

        return out

class Temporal_Attention_layer(nn.Module):
    '''
    Compute temporal attention scores for the graph based on input features over time.

    Parameters
    ----------
    in_channels: int
        The number of input channels (features per vertex).
    num_of_vertices: int
        The number of vertices (nodes) in the graph.
    num_of_timesteps: int
        The number of timesteps (time duration of input data).

    Attributes
    ----------
    U1, U2, U3: torch.nn.Parameter
        Learnable parameters for temporal attention computation.
    be: torch.nn.Parameter
        Bias for temporal attention computation.
    Ve: torch.nn.Parameter
        Learnable weight matrix for final temporal attention score computation.
    '''

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))

    def forward(self, x):
        '''
        Forward pass for temporal attention computation.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, N, F_in, T).

        Returns
        -------
        E_normalized: torch.Tensor
            Normalized temporal attention scores of shape (B, T, T), where:
            - B: batch size
            - T: number of timesteps
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # permute 后：(B, T, F_in, N)
        # x (B, T, F_in, N) @ U1 (N, 1)-> (B, T, F_in, 1)
        x_u1 = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
        # (B, T, F_in, 1) @ (F_in, N) -> lhs (B, T, F_in, N) 因为Fin是1，mad
        lhs = torch.matmul(x_u1, self.U2)

        # (F_in, ) @ (B, N, F_in, T)-> (B, N, T)
        # matmul是这样算的：先在第一个向量前增加一个维度1，乘完再去掉这个维度
        rhs = torch.matmul(self.U3, x)

        product = torch.matmul(lhs, rhs)  # (B, T, N) @ (B, N, T) -> (B, T, T)

        # 加权、加偏置
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)  # Apply softmax to normalize the attention scores
        return E_normalized


class ASTGCN_block(nn.Module):
    '''
    ASTGCN block: Combines spatial and temporal attention layers with Chebyshev graph convolution.

    Parameters
    ----------
    L_tilde: torch.Tensor
        Scaled Laplacian matrix of the graph.
    in_channels: int
        The number of input channels (features per vertex).
    K: int
        The order of the Chebyshev polynomial.
    nb_chev_filter: int
        The number of Chebyshev filters.
    nb_time_filter: int
        The number of time filters.
    time_strides: int
        The stride size for the time dimension.
    num_of_vertices: int
        The number of vertices (nodes) in the graph.
    num_of_timesteps: int
        The number of timesteps (time duration of input data).

    Attributes
    ----------
    TAt: Temporal_Attention_layer
        Temporal attention layer.
    SAt: Spatial_Attention_layer
        Spatial attention layer.
    cheb_conv_SAt: cheb_conv_withSAt
        Chebyshev graph convolution with spatial attention.
    time_conv: nn.Conv2d
        Convolutional layer for temporal dimension.
    residual_conv: nn.Conv2d
        Residual convolutional layer for input signal.
    ln: nn.LayerNorm
        Layer normalization applied to the output.
    '''

    def __init__(self, L_tilde, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices,
                 num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K = K, in_channels=in_channels,
                                               out_channels=nb_chev_filter, L_tilde=L_tilde)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

        # self.linear_substitute = nn.Linear(in_channels, nb_chev_filter)

    def forward(self, x):
        '''
        Forward pass for ASTGCN block.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, N, F_in, T), where:
            - batch_size: number of batches
            - N: number of vertices (nodes)
            - F_in: number of input features
            - T: number of timesteps

        Returns
        -------
        x_residual: torch.Tensor
            Output tensor of shape (batch_size, N, F_out, T), where:
            - F_out: number of output features (after convolution and attention)
            - T: number of timesteps
        '''

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # x: [b, N, F_in, T]

        # 1) Temporal Attention Layer (TAt)
        # (B, T, T) - temporal attention scores
        # temporal_At = self.TAt(x)
        # # (B, N, T) - 时间维度上加权
        # x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)
        # # 恢复x的形状
        # x_TAt = x_TAt.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # 2) Spatial Attention Layer (SAt)
        # (B, N, N) - spatial attention scores
        # spatial_At = self.SAt(x_TAt)

        spatial_At = torch.ones(batch_size, num_of_vertices, num_of_vertices, device=x.device)
        # Chebyshev Graph Convolution with Spatial Attention (cheb_conv_SAt)
        # (b, N, F_in, T) -> (b, N, F_out, T)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # Temporal Convolution (time_conv)
        # (b, F_out, N, T) -> (b, F_out, N, T)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))

        # Residual Convolution (residual_conv)
        # Residual (b, N, F_in, T) -> (b, F_out, N, T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))

        # Final output: Apply LayerNorm and return the processed result
        output = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return output

class ASTGCN_submodule(nn.Module):
    '''
    Submodule for ASTGCN: It stacks multiple ASTGCN blocks and adds a final convolution layer for prediction.

    Parameters
    ----------
    L_tilde: torch.Tensor
        Scaled Laplacian matrix of the graph.
    nb_block: int
        The number of ASTGCN blocks in this submodule.
    in_channels: int
        The number of input channels (features per vertex).
    K: int
        The order of the Chebyshev polynomial.
    nb_chev_filter: int
        The number of Chebyshev filters.
    nb_time_filter: int
        The number of time filters.
    time_strides: int
        The stride size for the time dimension.
    num_for_predict: int
        The number of timesteps for prediction.
    len_input: int
        The length of the input sequence.
    num_of_vertices: int
        The number of vertices (nodes) in the graph.

    Attributes
    ----------
    BlockList: nn.ModuleList
        List of ASTGCN blocks to process the input data.
    final_conv: nn.Conv2d
        Final convolutional layer to produce predictions from the processed output.
    '''

    def __init__(self, L_tilde, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 num_for_predict, len_input, num_of_vertices):
        super(ASTGCN_submodule, self).__init__()

        # Initialize the list of ASTGCN blocks
        self.BlockList = nn.ModuleList([ASTGCN_block(L_tilde, in_channels, K, nb_chev_filter, nb_time_filter,
                                                     time_strides, num_of_vertices, len_input)])
        self.BlockList.extend([ASTGCN_block(L_tilde, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                                            num_of_vertices, len_input // time_strides) for _ in
                               range(nb_block - 1)])

        # Final convolution to reduce the output channels for prediction
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

    def forward(self, x):
        '''
        Forward pass for ASTGCN submodule.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, N_nodes, F_in, T_in), where:
            - B: batch size
            - N_nodes: number of nodes in the graph
            - F_in: number of input features
            - T_in: number of timesteps

        Returns
        -------
        output: torch.Tensor
            Final output tensor of shape (B, N_nodes, T_out), where:
            - B: batch size
            - N_nodes: number of nodes in the graph
            - T_out: number of timesteps for prediction
        '''
        # Process the input through all ASTGCN blocks
        for block in self.BlockList:
            x = block(x)

        # Final prediction using convolution
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output

class ASTGCN(STBase):
    '''
    ASTGCN model: Combines multiple submodules (hour, day, week) for spatio-temporal prediction.

    Parameters
    ----------
    all_backbones: list[list]
        List of dictionaries for each submodule, defining parameters for hour, day, and week submodules.
    adj_mx: np.ndarray
        Adjacency matrix for the graph, shape (N, N), where N is the number of vertices.
    in_len: tuple
        Length of the input sequence, e.g. (3, 0) for recent data and period data.
    pred_len: int
        Length of the prediction sequence (forecasting steps).

    Attributes
    ----------
    submodules: nn.ModuleList
        List of submodules for different time periods (hour, day, week).
    fusion_weights: nn.Parameter
        Learnable parameters for fusing outputs from different submodules.
    '''

    def __init__(self, all_backbones, adj_mx,
                 in_len: int = 6,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1,
                 nb_block: int = 1,
                 **kwargs):
        super(ASTGCN, self).__init__(**kwargs)

        # Check input backbone lengths
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones must be greater than 0")

        in_channels = 1

        self.pred_len = pred_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.seq_len = in_len + period_len + trend_len

        num_of_vertices = adj_mx.shape[0]

        L_tilde = scaled_Laplacian(adj_mx) # grid graph
        # L_tilde = np.diag([1] * num_of_vertices)

        # Initialize submodules for hour, day, week
        self.submodules = []
        for backbones in all_backbones:
            nb_chev_filter = backbones['nb_chev_filter']
            nb_time_filter = backbones['nb_time_filter']
            time_strides = backbones['time_strides']
            in_len = backbones['in_len']
            K = backbones['K']
            self.submodules.append(
                ASTGCN_submodule(L_tilde, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                                 time_strides, pred_len, in_len, num_of_vertices))

        self.submodules = nn.ModuleList(self.submodules)
        self.fusion_weights = nn.Parameter(torch.Tensor(len(all_backbones), 1, num_of_vertices, pred_len))

        # Initialize parameters
        self.save_hyperparameters()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x_list):
        '''
        Forward pass for ASTGCN model.

        Parameters
        ----------
        x_list: list[mx.ndarray]
            List of input tensors, shape (batch_size, num_of_vertices, num_of_features, num_of_timesteps).
            e.g., Input 0 shape = torch.Size([32, 400, 1, 32])

        Returns
        -------
        out: torch.Tensor
            Output tensor after fusion, shape (B, N_nodes, T_out).
        '''
        # Check consistency of input shapes
        if len(x_list) != len(self.submodules):
            raise ValueError("Number of submodules does not match length of input list.")

        # Check number of vertices consistency
        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! Check input data size.")

        # Check batch size consistency
        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have the same batch size!")

        # Process through each submodule and perform fusion
        submodule_outputs = [self.submodules[idx](x_list[idx]) for idx in range(len(x_list))]
        out = torch.sum(self.fusion_weights * torch.stack(submodule_outputs), dim=0)
        out = rearrange(out, 'b n t -> b t n')
        return out

def scaled_Laplacian(W):
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
    D = np.diag(np.sum(W, axis=1)) # 因为用的是 grid_graph, 故不会有孤立节点问题
    L = D - W
    L = L.astype(np.float32)

    # Compute the largest eigenvalue for scaling
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
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
