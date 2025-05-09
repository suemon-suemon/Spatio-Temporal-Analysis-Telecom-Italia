import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import io
from PIL import Image
import pandas as pd
from utils.funcs import *
from models.modules import *
from models.STBase import STBase
from matplotlib import pyplot as plt


class GLLowPass(STBase):
    def __init__(self,
                 num_nodes: int = 400, # 节点数
                 feature_size: int = 6, # 时隙数，i.e.输入特征维度
                 if_temporal_graph: bool = False, # 是否学习时域图
                 *args, **kwargs,
                 ):
        """
        参数：
            num_nodes: 节点总数 N
            embed_size: 嵌入维度 F（将每个节点的 T 维输入映射到 F 维）
            feature_size: 输入特征维度 T
        """
        super(GLLowPass, self).__init__(*args, **kwargs)

        self.if_temporal_graph = if_temporal_graph

        if self.if_temporal_graph: # 时间图
            self.num_nodes = feature_size
            self.feature_size = num_nodes
            self.threshold = 0.005
            self.order = 20
            fraction_zero = 0.995  # 10% 的元素初始化为1
        else: # 空间图
            self.num_nodes = num_nodes
            self.feature_size = feature_size
            self.threshold = 0.6  # 空间图可以更稀疏
            self.order = 3
            fraction_zero = 0.2  # 20% 的元素初始化为1

        # 可学习的邻接权重向量 (上三角部分, 大小为 N*(N-1)/2)
        n_edges = self.num_nodes * (self.num_nodes - 1) // 2
        # self.theta = nn.Parameter(torch.randn(n_edges))

        theta_init = (torch.rand(n_edges) < fraction_zero).float()  # 生成0/1掩码
        self.theta = nn.Parameter(theta_init) # [0, 1]
        # softplus(x)=log(1+exp(x))
        self.w = F.softplus(self.theta) - 0.69  # [0.69, 1.31] - 0.6

        # 将 alpha1, alpha2, alpha3 作为可学习参数
        alpha_init = np.ones(self.order)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # 配合STBase，必须有这两个参数，虽然模型里不需要
        self.seq_len = feature_size
        self.pred_len = feature_size
        self.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    def vector_to_adj(self, dropout_rate = 0.0):
        """
        将参数向量 self.w（上三角部分）转换成全对称邻接矩阵 A (形状 [N, N])，
        并将对角线置为0
        """
        N = self.num_nodes
        # 归一化：确保 w 非零且和为固定值N/2
        self.w = F.softplus(self.theta) # 用theta参数化w, softplus(x)=log(1+exp(x))
        self.w = F.softshrink(self.w, lambd=self.threshold * self.w.detach().max())
        self.w = (self.w / (torch.sum(self.w) + 1e-8)) * (N/2)
        self.w = F.dropout(self.w, p=dropout_rate, training=self.training)

        A = torch.zeros(N, N, device=self.w.device)
        triu_indices = torch.triu_indices(N, N, offset=1)
        A[triu_indices[0], triu_indices[1]] = self.w
        A = A + A.t()
        self.A = A
        return A

    def compute_laplacian(self, A):
        """
        计算组合拉普拉斯矩阵 L = D - A
        """
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        return L

    def _log_intermediate_results(self, X):
        # X = X.to(self.device)
        wandb_logger = self.logger.experiment  # ✅ 使用 Lightning 传入的 wandb
        # results = self._compute_forward_results(X)
        # para_adj, adj_matrix_sampled = results['para_adj'], results['adj_matrix_sampled']

        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        A = self.vector_to_adj()
        # 将矩阵转换为 Pandas DataFrame 并存储为 CSV
        pd.DataFrame(A.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "A.csv"), index=False)

        # 计算节点度向量：对每行求和（因为矩阵是对称的，按行或列相同）
        # degrees = A.sum(dim=1).detach().cpu().numpy()
        degrees = (A != 0).sum(dim=1)

        # 创建两个子图：上图高度6，下图高度3，总高度9，宽度6。
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9),
                                       gridspec_kw={'height_ratios': [2, 1]})

        # 上图：显示邻接矩阵 A 的热图
        im1 = ax1.imshow(A.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        ax1.set_title("Adjacency Matrix A")
        ax1.set_xlabel("Nodes")
        ax1.set_ylabel("Nodes")
        plt.colorbar(im1, ax=ax1)

        # 下图：绘制度分布直方图
        ax2.hist(degrees.cpu().numpy(), bins=20, color='skyblue', edgecolor='black')
        ax2.set_title("Degree Distribution of A")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Number of nodes")

        plt.tight_layout()

        # 存储至本地
        a_ext_img_path = os.path.join(save_dir, "LearnedAdj.png")
        fig.savefig(a_ext_img_path, format='png')
        # 存储至wandb
        img_buf_A = io.BytesIO()
        plt.savefig(img_buf_A, format='png')
        img_buf_A.seek(0)
        img_A_pil = Image.open(img_buf_A)  # ✅ 这一步转换
        plt.close(fig)

        wandb_logger.log({
            "ImResult/VAE learned Adj (A)": wandb.Image(img_A_pil),
        })
        return

    def _compute_forward_results(self, x):
        """
        输入 x: [B, N, T]
        输出 X_recons: [B, N, embed_size]
        计算流程：
            1. 嵌入：X_prime = emb(x) [B, N, embed_size]
            2. 根据可学习向量 w 得到邻接矩阵 A，并计算拉普拉斯矩阵 L
            3. 计算图卷积：alpha1 * L X' + alpha2 * L^2 X' + alpha3 * L^3 X'
            4. 最终输出: X_recons = (图卷积结果) + X_prime   （跳跃连接）
        """
        # x: [B, N, T]
        if self.if_temporal_graph:
            x = x.permute(0, 2, 1) # [B, T, N]

        # B, num_nodes, feature_size = x.shape
        # if_temporal_graph: N = T, F = N
        # if not if_temporal_graph: N = N, F = T

        # 2. 邻接矩阵和拉普拉斯矩阵
        A = self.vector_to_adj()  # [N, N]
        L = self.compute_laplacian(A)  # [N, N]

        # 3. 计算多阶拉普拉斯乘积，使用递归结构
        alpha_prob = F.softmax(self.alpha, dim=0)

        conv_terms = []
        term = torch.matmul(L, x)  # term1, shape: [B, N, F]
        conv_terms.append(term)
        # 依次计算 L^k * X_prime, k=2,...,order
        for i in range(1, self.order):
            term = torch.matmul(L, term)  # 计算：term = L * (previous term)
            conv_terms.append(term)

        # 加权求和 sum_{i=1}^{order} alpha_prob[i]*term_i
        graph_conv_output = sum(alpha_prob[i] * conv_terms[i] for i in range(self.order))

        # 4. 跳跃连接: 加上原始输入 X_prime
        x_recons = graph_conv_output + x  # [B, N, feature_size]

        if self.if_temporal_graph:
            x_recons = x_recons.permute(0, 2, 1) # [B, feature_size, N]

        return x_recons

    def forward(self, X):
        X_recons = self._compute_forward_results(X)
        return X_recons


# 示例使用代码
if __name__ == '__main__':
    # 假设输入数据 x: [B, N, T]
    B, N, T = 8, 20, 10
    x = torch.randn(B, N, T, device='cuda')

    model = GLLowPass(num_nodes=N, feature_size=T,
                      if_temporal_graph=True,
                      )
    # 获得输出：X_recons 的形状为 [B, N, embed_size]
    X_recons = model(x)
    print("Output shape:", X_recons.shape)
    print("adj shape: ", model.A.shape)
