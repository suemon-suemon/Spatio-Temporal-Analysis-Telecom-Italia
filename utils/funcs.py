import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def AdjMat2LapMat(A, normalized=False):
    """
    A: [n,n] 的邻接矩阵 (torch.FloatTensor 或 torch.DoubleTensor 等)

    返回拉普拉斯矩阵 L
    """

    # 如果是 csr-array，转换为 numpy 数组
    if isinstance(A, csr_matrix):
        A =  A.toarray()
    # 输入数据转化为 tensor
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float)

    # A = A.to(torch.float32)  # 确保类型一致
    # 度向量 d
    d = A.sum(dim=-1)  # [n]

    if not normalized:
        # L = D - A
        D = torch.diag(d)
        L = D - A
    else:
        # 对称归一化拉普拉斯: L = I - D^{-1/2} * A * D^{-1/2}
        d_sqrt_inv = 1.0 / torch.sqrt(d + 1e-8)
        D_sqrt_inv = torch.diag(d_sqrt_inv)

        L = torch.eye(A.size(0), device=A.device, dtype=A.dtype) - D_sqrt_inv @ A @ D_sqrt_inv

    return L

def KNNGraph(X,
             k_neighbors: int = 4,
             radius: float = 2.0,
             return_adjacency_matrix: bool = True):
    """
    输入:
    X: 张量，形状为 (batch_size, N, T)，表示每个批次的 N 个节点和每个节点的 T 维特征。

    输出:
    edge_index 或 adjacency_matrix:
        - 如果 return_adjacency_matrix=True，返回邻接矩阵，形状为 (N, N)；
        - 如果 return_adjacency_matrix=False，返回边索引，形状为 (2, num_edges)，表示邻接矩阵的稀疏形式。
    """
    # 输入数据转化为 tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    batch_size, N, T = X.shape

    # Step 1: 将不同 batch 的数据串联起来，变成 (N, T * batch_size)
    X_flat = X.permute(1, 2, 0).reshape(N, T * batch_size).cpu().detach().numpy()

    # Step 2: 使用 KNN 算法计算邻接关系
    nbrs = NearestNeighbors(#n_neighbors=k_neighbors,
                            radius = radius,
                            algorithm="auto").fit(X_flat)
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

    if return_adjacency_matrix:
        return adjacency_matrix
    else:
        return edge_index

def normalize_adj(A):
    """
    normalisation of adjacency matrix
    Args:
        A: adj mat with/without self loops [N, N]
    Returns:
        A_normalize = D^{-1/2} A D^{-1/2}  [N, N]
    """
    row_sum = A.sum(-1)  # 对每一行求和，得到节点的度
    row_sum = (row_sum) ** -0.5  # 对度的平方根取倒数
    D = torch.diag(row_sum)  # 生成度矩阵 D
    A_normalized = torch.mm(torch.mm(D, A), D)  # 归一化邻接矩阵

    return A_normalized


def normalize_adj_add_self_hoop(A):
    """
    normalisation of adjacency matrix added self hoops (used for GCN)
    Args:
        A_hat: adj mat without self loops [N, N]
    Returns:
        A_normalize = D^{-1/2} (A+I) D^{-1/2}  [N, N]
    """
    # 额外检查是否存在 Inf 或 NaN
    if torch.isinf(A).any() or torch.isnan(A).any():
        print("Warning: Inf or NaN detected in unnormalized adjacency matrix!")

    # Remove all initial self-loops by setting the diagonal elements to zero
    A = A.clone()
    A.fill_diagonal_(0)

    # Add self loops (unit diagonal matrix)
    I = torch.eye(A.size(0), device = A.device)  # Create identity matrix with the same size as A_hat
    A = A + I  # Add self loops

    # Calculate the degree vector (sum of rows)
    row_sum = torch.sum(A, dim=1)  # shape: (N,)
    row_sum[row_sum < 1e-6] = 1  # 将孤立节点的度置为1

    d_inv_sqrt = row_sum.pow(-0.5) # 计算每个节点度的逆平方根
    d_inv_sqrt = torch.clamp(d_inv_sqrt, min=1e-3, max=1e3)  # Prevent extreme values
    D_inv_sqrt = torch.diag(d_inv_sqrt)  # Construct diagonal matrix

    # Compute normalized adjacency matrix
    A_normalized = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)

    if torch.isinf(A_normalized).any() or torch.isnan(A_normalized).any():
        print("Warning: Inf or NaN detected in normalized adjacency matrix!")

    return A_normalized


def sample_gumbel_from_uniform(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape) # [0,1)（包括 0，不包括 1）
    if torch.cuda.is_available():
        U = U.cuda()
    # Gumbel 分布可以通过对均匀分布采样后进行两次对数变换获得
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sample(logits, noise_sample):
    """Draw a sample from the Gumbel-Softmax distribution"""
    assert logits.shape == noise_sample.shape

    # No noise added to self loops so zero out those indices
    zero_self_loops = 1 - torch.eye(n=logits.shape[0])

    if torch.cuda.is_available():
        zero_self_loops = zero_self_loops.cuda()

    zero_self_loops = 1 - torch.eye(n=logits.shape[0])  # 避免自环
    noise = noise_sample * zero_self_loops
    y = logits + noise_sample
    return y


def straight_through_gumbel_softmax(logits, temperature, hard=False, self_loops_noise=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: input log probabilities of size [N, N]
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        Sample from the Gumbel-Softmax distribution. If hard=True, it will
        be one-hot during forward pass and differentiable during backward pass.
    """

    noise_sample = sample_gumbel_from_uniform(logits.size()).to(logits.device)
    y = gumbel_sample(logits, noise_sample)
    y = F.softmax(y / temperature, dim=-1)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def Adj2EdgeList(adjacency_matrix,
                 filter_small_values: bool = False,
                 print_warning: bool = True,
                 if_weight: bool = False):
    """
    将邻接矩阵转换为边索引列表。

    该函数将给定的邻接矩阵转换为一个边索引张量，其中每一列表示一条边的两个节点索引。
    若 if_weight 为 True，还会返回一个边权张量，其中包含每条边对应的权值。

    参数：
        adjacency_matrix (torch.Tensor or np.ndarray):
            一个形状为 (N, N) 的邻接矩阵，其中 N 是节点数量。
            矩阵的每个元素表示节点之间的连接（0表示无连接，非零值表示连接）。
        filter_small_values (bool): 是否滤除小于阈值的数值，默认为 False。
        print_warning (bool): 若邻接矩阵中包含 0 和 1 之外的值，则是否打印警告信息，默认为 True。
        if_weight (bool): 如果为 True，则返回 (edge_index, edge_weight)；否则只返回 edge_index。

    返回：
        当 if_weight 为 False 时，返回：
            torch.Tensor: 形状为 (2, num_edges) 的边索引张量，dtype 为 long。
        当 if_weight 为 True 时，返回：
            (edge_index, edge_weight)
            edge_index 是形状为 (2, num_edges) 的张量 (dtype long)；
            edge_weight 是形状为 (num_edges,) 的张量，包含对应边的权值。

    示例：
        输入：
            adjacency_matrix =
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
        若 if_weight=False，则输出 edge_index =
            [[0, 1, 1],
             [1, 2, 0]]
        若 if_weight=True，则输出 (edge_index, edge_weight)，其中 edge_weight 为 [1, 1, 1]。
    """
    # 如果是 csr-array，转换为 numpy 数组
    from scipy.sparse import csr_matrix  # 如果尚未导入
    if isinstance(adjacency_matrix, csr_matrix):
        adjacency_matrix = adjacency_matrix.toarray()

    # 输入数据转化为 tensor
    if not isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = torch.tensor(adjacency_matrix)

    # 是否去除小值
    if filter_small_values:
        # 假设 threshold_small 是已定义的函数，用于将小于阈值的元素置零
        adjacency_matrix = threshold_small(adjacency_matrix, threshold=1e-3)

    # 检查矩阵是否包含 0 和 1 之外的值
    if print_warning:
        if not torch.all(torch.logical_or(adjacency_matrix == 0, adjacency_matrix == 1)):
            print(
                "Warning: adjacency matrix contains values other than 0 or 1. This operation may not be differentiable!")

    # 找到邻接矩阵中非零元素的索引，得到形状 [2, num_edges] 的张量
    edges = (adjacency_matrix > 0).nonzero().t().long()

    # 如果需要返回边权，则从邻接矩阵中取出对应的权值
    if if_weight:
        # edges 的第一行和第二行分别为起始节点和终止节点索引
        edge_weight = adjacency_matrix[edges[0], edges[1]]
        return edges, edge_weight
    else:
        return edges


def GauSamplesGenerate(mu, logvar):
    # logvar不能太大或太小
    # 限制 logvar 的范围，避免 exp(·) 计算溢出
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * F.softplus(logvar))  # 更稳定
    eps = torch.randn_like(std)    # 生成标准正态噪声 ε
    return mu + std * eps          # 采样 z


def BernGauSamplesGenerate(theta, mu, logvar, temperature = 0.5, clamp_strength=10):

    # 生成与 theta 同形状的均匀随机数，范围[0,1],避免太大或太小
    eps = torch.rand_like(theta).clamp(min=1e-6, max=1 - 1e-6)

    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * F.softplus(logvar))  # 更稳定
    gausamples =  mu + std * eps  # 采样 z

    # 使用 Sigmoid 进行软裁剪，使 theta 避免精确等于 0 或 1，0->0.0045,1->0.9955
    theta = torch.sigmoid(clamp_strength * (theta - 0.5))  # 软裁剪

    # 使用 Gumbel噪声： -log(-log(eps))
    gumbel_noise = -torch.log(-torch.log(eps))
    # 将伯努利概率 \theta 转换成 logits：
    logit = torch.log(theta) - torch.log(1 - theta)
    # 通过加上噪声后除以温度，再经过 sigmoid 得到采样结果
    bernsamples = torch.sigmoid((logit + gumbel_noise) / temperature)

    samples = bernsamples * gausamples

    return samples

def BernSamplesGenerate(theta, temperature, clamp_strength=10):
    """
    输入:
        theta: 伯努利概率张量，形状为 [batch, ...]
        temperature: 温度低，则0/1值多；温度高，则0～1之间的数值多
    输出:
        使用重参数化技巧采样得到的伯努利张量
        与Gumbel-Softmax的区别：
            1）logits是用伯努利分布参数theta生成的
            2）不用softmax，而使用sigmoid，因为是二分类。但本质上还是一样的。
    """
    # 使用 Sigmoid 进行软裁剪，使 theta 避免精确等于 0 或 1，0->0.0045,1->0.9955
    theta = torch.sigmoid(clamp_strength * (theta - 0.5))  # 软裁剪
    # 生成与 theta 同形状的均匀随机数，范围[0,1],避免太大或太小
    eps = torch.rand_like(theta).clamp(min=1e-6, max=1-1e-6)
    # 使用 Gumbel噪声： -log(-log(eps))
    gumbel_noise = -torch.log(-torch.log(eps))
    # 将伯努利概率 \theta 转换成 logits：
    logit = torch.log(theta) - torch.log(1 - theta)
    # 通过加上噪声后除以温度，再经过 sigmoid 得到采样结果
    samples = torch.sigmoid((logit + gumbel_noise) / temperature)

    return samples


def BernKLDiv(p, prior = 0.5, eps = 1e-20):

    # 计算“输入伯努利分布”与“理想伯努利分布”之间的KL散度，作为Bern-VAE的隐空间正则化损失项
    # p: 输入伯努利分布的参数，tensor
    # prior: 先验概率（默认 0.5，即理想伯努利分布均值），此数值越小，越提升所习伯努利分布的稀疏程度

    # 将 prior 转换为和 input 相同类型和设备的张量
    prior_tensor = torch.tensor(prior, device=p.device, dtype=p.dtype)

    # 计算每个元素的 KL 散度： p*log(p/prior) + (1-p)*log((1-p)/(1-prior))
    KLDiv = p * (torch.log(p + eps) - torch.log(prior_tensor)) + \
           (1 - p) * (torch.log(1 - p + eps) - torch.log(1 - prior_tensor))

    return KLDiv.mean()


# 硬阈值操作，将极小数字置零
def threshold_small(x, threshold=1e-5):
    # 创建一个与 x 同设备、同数据类型的0张量
    zero_tensor = torch.zeros_like(x)
    # 当 |x| < threshold 时，置为0，否则保持原值
    return torch.where(torch.abs(x) < threshold, zero_tensor, x)



############################# Some Abandoned Codes ###################################
class GCNConv_mine(nn.Module):
    # 自己写的就是不如人家自带的好使啊
    def __init__(self, in_dim, out_dim):
        super(GCNConv_mine, self).__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        # 使用 He 初始化权重矩阵, 适用于 ReLU 激活函数
        nn.init.kaiming_uniform_(self.W, nonlinearity='relu')
        # 如果使用 sigmoid 或 tanh 激活函数，则使用 Xavier 初始化
        # nn.init.xavier_uniform_(self.W)

    def forward(self, x, adj):
        """
        Args:
            x: input tensor, shape: [batch_size, num_nodes, in_dim]
            adj: normalized adjacency matrix, shape: [num_nodes, num_nodes]
        Returns:
            out: output tensor after GCN layer, shape: [batch_size, num_nodes, out_dim]
        """
        # 计算图卷积: AxW
        Ax = torch.matmul(adj, x)  # 矩阵乘法: 计算邻接矩阵与特征矩阵的乘积
        AxW = torch.matmul(Ax, self.W)  # 再与权重矩阵相乘
        out = F.leaky_relu(AxW, negative_slope=0.2)  # ReLU 激活函数
        # out = self.layer_norm(sigmaAxW)  # 添加 LayerNorm 层来稳定数值范围
        return out


