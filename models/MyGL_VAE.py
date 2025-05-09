import wandb
import os
import io
from PIL import Image
import pandas as pd
from utils.funcs import *
from models.modules import *
from models.STBase import STBase
from matplotlib import pyplot as plt

class BernEncoder(nn.Module):
    def __init__(self, in_dim, N):
        """
        Args:
            in_dim (int): 输入的原始特征维度 T
            N (int): 节点数
        """
        super(BernEncoder, self).__init__()
        self.N = N
        self.in_dim = in_dim

        self.linear1 = nn.Linear(in_dim, in_dim * 8)
        self.linear2 = nn.Linear(in_dim, in_dim * 8)
        self.linear_para_latent = nn.Linear(in_dim * 16, in_dim * 8)
        self.linear_para_Bern = nn.Linear(in_dim * 8, 1)
        self.latent_dim = in_dim * 8
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

        return para_Bern


class Encoder(nn.Module):
    """编码器网络，将节点特征编码为隐变量"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """解码器网络，从邻接矩阵生成重建数据"""
    def __init__(self, recon_dim, num_nodes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(num_nodes, recon_dim * 16)
        self.fc2 = nn.Linear(recon_dim * 16, recon_dim * 32)
        self.fc3 = nn.Linear(recon_dim * 32, recon_dim * 16)
        self.fc4 = nn.Linear(recon_dim * 16, recon_dim)

    def forward(self, z):
        # [N, N] -> [N, T]
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = self.fc4(z)
        return z

class GraphLearnVAE(STBase):
    """整体模型：整合编码器和解码器模块"""
    def __init__(self,
                 N: int = 400, # 节点数
                 input_time_steps: int = 6, # 每个节点的特征数（时隙数）
                 sym_sample = True, # 采样结果是否对称化
                 lambda_smooth: float = 0.0001, # 图平滑项损失权重
                 lambda_sparse: float = 0.0000001, # 图稀疏项损失权重
                 *args, **kwargs):
        super(GraphLearnVAE, self).__init__(*args, **kwargs)
        self.N = N
        self.sym_sample = sym_sample # 采样结果是否对称化
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.encoder = BernEncoder(in_dim = input_time_steps, N = N)
        self.decoder = Decoder(recon_dim = input_time_steps, num_nodes = N)

        self.seq_len = input_time_steps
        self.pred_len = input_time_steps

    def reparameterize(self, para_Bern):
        adj_matrix_sampled = BernSamplesGenerate(para_Bern, temperature=0.2, clamp_strength=10)
        if self.sym_sample:
            adj_matrix_sampled = (adj_matrix_sampled + adj_matrix_sampled.t()) / 2
        return adj_matrix_sampled

    def _compute_forward_results(self, X):

        batch_size, N, input_time_steps = X.size()
        # 编码器
        para_adj = self.encoder(X)
        # 重参数化
        adj_matrix_sampled = self.reparameterize(para_adj) # [N, N]
        # 解码器生成重建数据
        adj_matrix_expand = adj_matrix_sampled.expand(batch_size, -1, -1) # [B, N, N]
        x_decoded = self.decoder(adj_matrix_expand)

        results = {
            'para_adj': para_adj,
            'adj_matrix_sampled': adj_matrix_sampled,
            'x_decoded': x_decoded,
        }
        return results

    def forward(self, X):
        results = self._compute_forward_results(X)
        L = AdjMat2LapMat(results['adj_matrix_sampled'])
        loss_smooth = (self.lambda_smooth *
                       torch.mean(torch.einsum('bit,ij,bjt->b', X, L, X)))
        loss_sparse = self.lambda_sparse * torch.norm(results['adj_matrix_sampled'], p=1)
        return results['x_decoded'], loss_smooth + loss_sparse

    def _log_intermediate_results(self, X):
        X = X.to(self.device)
        wandb_logger = self.logger.experiment  # ✅ 使用 Lightning 传入的 wandb
        results = self._compute_forward_results(X)
        para_adj, adj_matrix_sampled = results['para_adj'], results['adj_matrix_sampled']

        save_dir = os.path.join(self.result_dir, str(self.logger.version))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 将矩阵转换为 Pandas DataFrame 并存储为 CSV
        pd.DataFrame(para_adj.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "para_adj.csv"), index=False)
        pd.DataFrame(adj_matrix_sampled.detach().cpu().numpy()).to_csv(os.path.join(save_dir, "adj_matrix_sampled.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列的子图
        # 第一个子图展示 para_adj
        im1 = axes[0].imshow(para_adj.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        axes[0].set_title("para_adj")
        axes[0].set_xlabel("Nodes")
        axes[0].set_ylabel("Nodes")
        plt.colorbar(im1, ax=axes[0])
        # 第二个子图展示 adj_matrix_sampled
        im2 = axes[1].imshow(adj_matrix_sampled.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        axes[1].set_title("adj_matrix_sampled")
        axes[1].set_xlabel("Nodes")
        axes[1].set_ylabel("Nodes")
        plt.colorbar(im2, ax=axes[1])
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



