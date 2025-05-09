import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import csv
import numpy as np
import networkx as nx
from utils.registry import register
from models.STBase import STBase


def load_spatialmatrix(dataset, num_node):
    """
    加载并处理空间矩阵（spatial matrix）

    参数:
        dataset: 数据集名称 ('PEMSD3', 'PEMSD4', 'PEMSD7', 'PEMSD8')
        num_node: 节点数量

    返回:
        sp_matrix: 归一化的空间邻接矩阵 [num_node, num_node]
    """
    files = {'PEMSD3': ['PeMSD3/pems03.npz', 'PeMSD3/distance.csv', 'pems03'],
             'PEMSD4': ['PeMSD4/pems04.npz', 'PeMSD4/distance.csv', 'pems04'],
             'PEMSD7': ['PeMSD7/pems07.npz', 'PeMSD7/distance.csv', 'pems07'],
             'PEMSD8': ['PeMSD8/pems08.npz', 'PeMSD8/distance.csv', 'pems08'], }
    filename = dataset
    file = files[filename]
    filepath = "../data/"

    # 如果处理过的距离矩阵不存在，则从CSV加载并处理
    if not os.path.exists(f'../data/{file[2]}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            file_csv = csv.reader(fp)
            for line in file_csv:
                break  # 跳过CSV头行
            for line in file_csv:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])  # 保持矩阵对称性
            np.save(f'../data/{file[2]}_spatial_distance.npy', dist_matrix)

    # 加载距离矩阵并归一化
    dist_matrix = np.load(f'../data/{file[2]}_spatial_distance.npy')
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std

    # 使用高斯核将距离转换为相似度
    sigma = 10  # 高斯核参数，空间矩阵为10，DTW矩阵为0.1
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix <= 0] = 0.
    sp_matrix[sp_matrix > 0] = 1.  # 二值化

    # 对称归一化（类似于GCN中的归一化方式）
    sp_matrix = normalize_spmatrix(sp_matrix)
    return sp_matrix

def normalize_spmatrix(sp_matrix):
    # 将邻接矩阵转换为 tensor 张量
    if not torch.is_tensor(sp_matrix):
        sp_matrix = torch.from_numpy(sp_matrix).to(torch.float32)

    # 对称归一化（类似于GCN中的归一化方式）
    ds = torch.sum(sp_matrix, dim=0)  # 度向量
    Ds = torch.diag(torch.rsqrt(ds))  # D^(-1/2)
    sp_matrix = torch.matmul(Ds, torch.matmul(sp_matrix, Ds))  # D^(-1/2) * A * D^(-1/2)
    return sp_matrix

def get_matrix_list(order, matrix):
    '''
    生成不同阶的空间或时间矩阵列表（bivariate Bernstein polynomial approximation, eq.15）

    参数:
        order: 空间或时间维度的近似阶数
        matrix: 空间或时间邻接矩阵

    返回:
        矩阵列表 [order, matrix_size, matrix_size]
    '''
    matrix_size = matrix.shape[0]  # 时间步长或节点数量
    L1 = torch.eye(matrix_size).to(matrix) + matrix
    L2 = torch.eye(matrix_size).to(matrix) - matrix
    matrix_list = []
    for i in range(order):
        # bivariate Bernstein polynomial approximation多项式系数（二项式系数）
        weight = (1 / 2 ** order) * (
                math.factorial(order) / (math.factorial(i) * math.factorial(order - i)))
        # 计算矩阵幂
        matrix = torch.mm(torch.matrix_power(L1, order - i), torch.matrix_power(L2, i))
        matrix_list.append(weight * matrix)
    return torch.stack(matrix_list, dim=0)


class STEmbedding(nn.Module):
    '''
    时空嵌入模块

    参数:
        s_order: 空间维度的近似阶数
        t_order: 时间维度的近似阶数
        num_nodes: 节点数量

    输入:
        TE: [batch_size, num_his, 2] (星期几, 一天中的时间)
        T: 一天中的时间步数

    返回:
        包含时空信息的系数集 [batch_size, t_order, s_order]
    '''

    def __init__(self, s_order, t_order, num_nodes, input_window):
        super(STEmbedding, self).__init__()

        # 随机初始化空间嵌入 [N, 10]
        self.SE = nn.Parameter(torch.FloatTensor(num_nodes, 10))
        # 也可以使用单位矩阵初始化: self.SE = nn.Parameter(torch.eye(num_nodes))

        # 时间嵌入MLP，12 = self.input_window
        self.tmlp1 = torch.nn.Conv1d(295, s_order, kernel_size=1, padding=0, bias=True)  # 295 = 7+288
        self.tmlp2 = torch.nn.Conv1d(input_window, t_order, kernel_size=1, padding=0, bias=True)

        # 空间嵌入MLP
        self.smlp1 = torch.nn.Conv1d(num_nodes, t_order, kernel_size=1, padding=0, bias=True)
        self.smlp2 = torch.nn.Conv1d(10, s_order, kernel_size=1, padding=0, bias=True)

        self.num_nodes = num_nodes

    def forward(self, TE, T=288):
        # TE [B, T, 2], 分别是一天中的星期几和一天中的时间
        # T=288表示一天分为288个时间片段（5分钟一个）

        # 对星期几进行one-hot编码 [B, T, 7]
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B T 7]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)

        # 对一天中的时间进行one-hot编码 [B, T, 288]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B T 288]
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)

        # 合并时间特征
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B, T, 288+7]

        # 通过MLP转换时间特征
        TE = F.relu(self.tmlp1(TE.permute(0, 2, 1)))  # [B, s_order, T]
        TE = F.relu(self.tmlp2(TE.permute(0, 2, 1)))  # [B, t_order, s_order]

        # 通过MLP转换空间特征, SE [num_nodes, 10], 没有 batch_size, 考虑各个批次的空间信息是一样的
        SE = F.relu(self.smlp1(self.SE))  # [t_order, num_nodes]
        SE = F.relu(self.smlp2(SE.permute(1, 0)).T)  # [t_order, s_order]

        # 时空特征融合(相加)
        STE = F.relu(SE + TE)  # [B, t_order, s_order]

        # 删除变量 dayofweek 和 timeofday，释放内存
        del dayofweek, timeofday
        return STE


class ST_Block(nn.Module):
    '''
    时空卷积模块

    参数:
        c_in: 输入通道数
        c_out: 输出通道数
        s_order: 空间维度的近似阶数
        t_order: 时间维度的近似阶数

    输入:
        x: [batch_size, c_in, num_nodes, time_step]
        sp_matrix: 空间邻接矩阵 [num_nodes, num_nodes]
        tp_matrix: 时间邻接矩阵 [time_steps, time_steps]
        STE: 时空嵌入 [batch_size, t_order, s_order]

    返回:
        hidden: [batch_size, c_out, num_nodes, time_step]
        theta_matrix: [batch_size, t_order, s_order]
    '''

    def __init__(self, c_in, c_out, dropout, s_order=10, t_order=5):
        super(ST_Block, self).__init__()
        self.c_in = c_in
        self.s_order = s_order
        self.t_order = t_order
        self.mlp = torch.nn.Conv2d(self.c_in * 2, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.theta_mlp1 = torch.nn.Conv1d(self.t_order, self.t_order, kernel_size=1, bias=True)
        self.theta_mlp2 = torch.nn.Conv1d(self.s_order, self.s_order, kernel_size=1, bias=True)

        self.dropout = dropout

    def forward(self, x, sp_matrix, tp_matrix, STE):
        # 获取不同阶数的时间和空间的bern多项式矩阵列表，时间图是 toeplitz 矩阵，空间图是位置邻接矩阵
        tp_matrix_list = get_matrix_list(self.t_order, tp_matrix).to(x.device)  # [t_order, time_step, time_step]
        sp_matrix_list = get_matrix_list(self.s_order, sp_matrix).to(x.device)  # [s_order, node_num, node_num]

        out = [x]  # 用于跳跃连接

        # 权重矩阵，空间-时间2D滤波器系数, 是由时空信息嵌入特征 STE 线性变换而得
        theta_matrix = F.relu(
            self.theta_mlp2(self.theta_mlp1(STE).permute(0, 2, 1)).permute(0, 2, 1)).to(x.device)
        # [batch_size, t_order, s_order]

        # 时空卷积操作 x_st = sp_matrix_list * theta * tp_matrix_list * x
        # 先与时间权重矩阵相乘
        tweight_list = torch.einsum('bts,tnm->bsnm', theta_matrix,
                                    tp_matrix_list)  # [batch_size, s_order, time_step, time_step]

        # print('x shape: ', x.shape)
        # print('tweight shape: ', tweight_list.shape)
        x = x.unsqueeze(1).expand(-1, self.s_order, -1, -1, -1) # x: bfnt -> bofnt
        x_t_list = torch.einsum('botk,bofnt->bofnt', tweight_list,
                                x)  # [batch_size, s_order, feature_dim, node_num, time_step]

        # 再与空间权重矩阵相乘
        x_st = torch.einsum('omn,bofnt->bfmt', sp_matrix_list,
                            x_t_list)  # [batch_size, feature_dim, node_num, time_step]

        # 跳跃连接和输出处理
        out.append(x_st)
        hidden = self.mlp(torch.cat(out, dim=1))  # 将原始输入和处理后的特征拼接
        hidden = self.bn(hidden)  # 批归一化
        hidden = F.dropout(hidden, self.dropout, training=self.training)  # Dropout正则化

        # 释放内存
        del tp_matrix_list, sp_matrix_list, tweight_list, x_t_list, x_st

        return hidden, theta_matrix


class PyrTempConv(nn.Module):
    """
    金字塔时间卷积模块，多尺度特征提取

    参数:
        c_in: 输入通道数
        stride: 步长
        kernel: 卷积核大小

    输入:
        x: [batch_size, c_in, num_nodes, time_step]

    返回:
        fusion: [batch_size, c_in, num_nodes, time_step]
    """

    def __init__(self, c_in, stride, kernel):
        super(PyrTempConv, self).__init__()
        self.c_in = c_in

        # 不同尺度的卷积层
        self.pyr1 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 1), stride=(1, 1))  # 1x1卷积
        self.pyr2 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 3), stride=(1, 3))  # 1x3卷积
        self.pyr3 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 6), stride=(1, 6))  # 1x6卷积

        # 融合层
        self.conv = nn.Sequential(
            nn.Conv2d(self.c_in * 3, self.c_in, kernel_size=1))  # 1x1卷积融合特征
        self.bn = nn.BatchNorm2d(self.c_in)  # 批归一化

    def forward(self, x):
        # GLU门控机制处理各尺度特征
        x1_gate, x1_filter = torch.split(self.pyr1(x), self.c_in, dim=1)
        x1 = torch.sigmoid(x1_gate) * torch.tanh(x1_filter)  # 门控线性单元(GLU)

        x2_gate, x2_filter = torch.split(self.pyr2(x), self.c_in, dim=1)
        x2 = torch.sigmoid(x2_gate) * torch.tanh(x2_filter)

        x3_gate, x3_filter = torch.split(self.pyr3(x), self.c_in, dim=1)
        x3 = torch.sigmoid(x3_gate) * torch.tanh(x3_filter)

        # 上采样到原始尺寸
        # 为啥没有 x1 上采样
        x2 = F.interpolate(x2, x.shape[2:], mode='bilinear')
        x3 = F.interpolate(x3, x.shape[2:], mode='bilinear')

        # 特征融合
        concat = torch.cat([x1, x2, x3], 1)  # 通道维度拼接
        fusion = self.bn(self.conv(concat))  # 卷积融合并批归一化
        return fusion

@register("STSGNN")
class STSGNN(STBase):
    """
    时空图神经网络模型(STSGNN), 时间空间两维图滤波，空间图grid，时间 topelitze 图
    """

    def __init__(self,
                 num_nodes: int = 400,
                 input_dim: int = 1,
                 horizon: int = 6,
                 output_window: int = 3,
                 output_dim: int = 1,
                 *args, **kwargs):
        super(STSGNN, self).__init__(*args, **kwargs)
        # self.dataset = dataset
        self.num_nodes = num_nodes  # 节点数
        self.feature_dim = input_dim  # 输入特征维度
        self.input_window = horizon  # 输入时间窗口大小
        self.output_window = output_window  # 输出时间窗口大小
        self.output_dim = output_dim  # 输出特征维度
        self.s_order = 12  # 12, 空间维度的近似阶数
        self.t_order = 6  # 6, 时间维度的近似阶数

        self.dropout = 0.3
        self.layers = 8  # 模型层数
        self.kernel = 2
        self.stride = 2 #kernel_size
        self.nhid = 32
        self.residual_channels = 32 # rnn_units, 隐藏层特征数
        self.skip_channels = self.nhid * 8  # nhid * 8, 跳跃连接通道数
        self.end_channels = self.nhid * 16  # nhid * 16, 末端卷积通道数

        self.seq_len = horizon
        self.pred_len = output_window
        # self.device = torch.device('cuda:0')

        # 加载空间邻接矩阵
        # sp_matrix = load_spatialmatrix(self.dataset, self.num_nodes)
        # # 对于KnowAir数据集，可以直接加载: sp_matrix = np.load('../data/KnowAir/knowair_adj_mat.npy')
        # self.sp_matrix = sp_matrix.to(self.device)
        G = nx.grid_2d_graph(20, 20)
        adj_mx = nx.adjacency_matrix(G)
        self.sp_matrix  = normalize_spmatrix(adj_mx.todense())

        # 时空嵌入层
        self.stembedding = STEmbedding(s_order=self.s_order, t_order=self.t_order,
                                       num_nodes=self.num_nodes, input_window=self.input_window)

        # 输入投影层
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        # 模型主体部分
        self.st_blocks = nn.ModuleList()  # 时空图卷积模块
        self.pry_blocks = nn.ModuleList()  # 金字塔时间卷积模块
        self.skip_convs = nn.ModuleList()  # 跳跃连接卷积层
        self.bn = nn.ModuleList()  # 批归一化层

        # 创建多层网络结构，共 self.layers = 8 层
        for i in range(self.layers):
            self.st_blocks.append(ST_Block(c_in=self.residual_channels, c_out=self.residual_channels,
                                           dropout=self.dropout, s_order=self.s_order, t_order=self.t_order))
            self.pry_blocks.append(PyrTempConv(c_in=self.residual_channels, stride=self.stride,
                                               kernel=self.kernel))
            self.skip_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(self.residual_channels))

        # 输出层
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

        # 创建时间邻接矩阵（12x12对角线加上下一/上一时间步的连接）
        tp_matrix = (F.pad(torch.eye(self.input_window-1), (0, 1, 1, 0), 'constant', 0) +
                     F.pad(torch.eye(self.input_window-1), (1, 0, 0, 1), 'constant', 0)).to(self.device)
        # 对时间邻接矩阵进行归一化
        dt = torch.sum(tp_matrix, dim=0)
        Dt = torch.diag(torch.rsqrt(dt))
        self.tp_matrix = torch.matmul(Dt, torch.matmul(tp_matrix, Dt)).to(self.device)  # D^(-1/2) * A * D^(-1/2)

    def forward(self, source): # targets, teacher_forcing_ratio=0.5
        """
        前向传播

        参数:
            source: 输入数据 [batch_size, input_window, num_nodes, feature_dim]
        返回:
            x: 预测结果 [batch_size, output_window, num_nodes, 1]
        """

        # source
        # [batch_size, input_window, num_nodes, traffic_data||day_of_week||time_of_day]

        # 分离特征
        inputs = source[:, :, :, 0:1]  # 交通流量特征 [batch_size, input_window, num_nodes, 1]
        temp = source[:, :, 1, 1:]  # 时间特征（星期几和一天中的时间） [batch_size, input_window, 2]

        # 获取时空嵌入
        STE = self.stembedding(temp)  # [batch_size, t_order, s_order]

        # 调整输入维度
        inputs = inputs.permute(0, 3, 2, 1)  # [batch_size, feature_dim, num_nodes, input_window]
        x = inputs

        # 开始卷积
        x = self.start_conv(x)  # [batch_size, residual_channels, num_nodes, input_window]
        skip = 0  # 跳跃连接初始化

        # 多层时空图卷积处理
        for i in range(self.layers):
            residual = x  # 保存残差连接

            # 时空图卷积
            x_st, theta_matrix = self.st_blocks[i](residual, self.sp_matrix, self.tp_matrix, STE)

            # 金字塔时间卷积
            x_st_pry = self.pry_blocks[i](x_st)

            # 处理跳跃连接
            s = x_st_pry
            s = self.skip_convs[i](s)
            skip = s + skip

            # 残差连接
            x = x_st_pry + residual
            x = self.bn[i](x)

        # 输出层处理
        x = F.relu(skip[:, :, :, -self.feature_dim:])  # 只取最后一个时间步 [batch_size, skip_channels, num_nodes, 1]
        x = F.relu(self.end_conv_1(x))  # [batch_size, end_channels, num_nodes, 1]
        x = self.end_conv_2(x)  # [batch_size, output_window, num_nodes, 1]

        return x
