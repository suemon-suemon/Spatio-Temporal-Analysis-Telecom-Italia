import torch
import torch.nn as nn
import torch.nn.functional as F
from models.STBase import STBase
from utils.registry import register

class GLU(nn.Module):
    """
    门控线性单元 (Gated Linear Unit)
    用于控制信息流动，类似于 LSTM 中的门控机制
    """

    def __init__(self, input_channel, output_channel):
        """
        初始化 GLU 模块

        参数:
            - input_channel (int): 输入特征的维度
            - output_channel (int): 输出特征的维度
        """
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)  # 左侧线性变换
        self.linear_right = nn.Linear(input_channel, output_channel)  # 右侧线性变换（门控部分）

    def forward(self, x):
        """
        前向传播

        参数:
            - x (Tensor): 输入张量，形状为 [batch_size, node_cnt, input_channel]

        返回:
            - output (Tensor): 输出张量，形状为 [batch_size, node_cnt, output_channel]
        """
        # 左侧线性变换与右侧通过 sigmoid 激活的线性变换的元素乘积
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    """
    预测模型的主要构建块
    结合了图卷积和频域处理的方法来捕获时间和空间依赖关系
    """

    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        """
        初始化 StockBlockLayer

        参数:
            - time_step (int): 时间步长，表示输入时间序列的长度
            - unit (int): 隐藏单元的数量
            - multi_layer (int): 多层因子，用于控制输出通道的大小
            - stack_cnt (int): 堆叠计数，表示当前块在模型中的位置
        """
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer

        # 图卷积的权重参数，形状为 [1, 4, 1, time_step*multi_layer, time_step*multi_layer]
        # 4 表示切比雪夫多项式的阶数（K+1，这里 K=3）
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))
        nn.init.xavier_normal_(self.weight)

        # 预测和重构的线性层
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)  # 预测中间层
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)  # 最终预测层

        # 只有在第一个堆叠块中才需要重构层
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)  # 重构层

        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)  # 短连接
        self.relu = nn.ReLU()

        # GLU 模块列表，用于频域处理
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi  # 输出通道数

        # 构建 GLU 层，共有 3 层，每层包含 2 个 GLU（分别处理实部和虚部）
        for i in range(3):
            if i == 0:
                # 第一层输入通道为 time_step * 4（4 是切比雪夫多项式的阶数）
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                # 中间层
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                # 最后一层
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        """
        频域处理单元，通过 FFT 变换和 GLU 处理序列数据

        参数:
            - input (Tensor): 输入张量，形状为 [batch_size, k, input_channel, node_cnt, time_step]
                              k 是切比雪夫多项式的阶数 (K+1)

        返回:
            - iffted (Tensor): 输出张量，形状为 [batch_size, 4, node_cnt, time_step]
        """
        batch_size, k, input_channel, node_cnt, time_step = input.size()

        # 重塑输入以进行 FFT
        input = input.view(batch_size, -1, node_cnt, time_step)  # [batch_size, k*input_channel, node_cnt, time_step]

        # 应用快速傅里叶变换
        ffted = torch.fft.fft(input, dim=-1)  # 保持原来的行为
        # 提取实部和虚部，并重塑为 [batch_size, node_cnt, k*input_channel*time_step]
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        # 通过 GLU 层处理实部和虚部
        for i in range(3):
            real = self.GLUs[i * 2](real)  # 处理实部
            img = self.GLUs[2 * i + 1](img)  # 处理虚部

        # 重塑为 [batch_size, 4, node_cnt, -1]
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()

        # 合并实部和虚部，准备逆 FFT
        time_step_as_inner = torch.complex(real, img)  # 不再拼成 [..., 2]，直接合成复数

        # 应用逆快速傅里叶变换
        iffted_complex = torch.fft.ifft(time_step_as_inner, dim=-1)  # 返回复数 Tensor
        iffted = iffted_complex.real  # 取实部作为时域信号

        return iffted

    def forward(self, x, mul_L):
        """
        前向传播

        参数:
            - x (Tensor): 输入张量，形状为 [batch_size, node_cnt, time_step]
            - mul_L (Tensor): 切比雪夫多项式转换后的拉普拉斯矩阵，形状为 [batch_size, k, node_cnt, node_cnt]
                             k 是切比雪夫多项式的阶数 (K+1)

        返回:
            - forecast (Tensor): 预测输出，形状为 [batch_size, node_cnt, horizon]
            - backcast_source (Tensor): 重构输出，形状为 [batch_size, node_cnt, time_step] 或 None
        """
        mul_L = mul_L.unsqueeze(1)  # [batch_size, 1, k, node_cnt, node_cnt]
        x = x.unsqueeze(1)  # [batch_size, 1, node_cnt, time_step]

        # 图卷积操作
        if self.stack_cnt == 0:
            x = x.permute(0, 1, 2, 4, 3)
        # print('mul_L shape: ', mul_L.shape, 'x shape: ', x.shape)

        # 这里的 GFT 是 4-order Chev GF，节点域，k = 4
        # 好奇怪，这不是 GFT 啊，这已经等价于在频域做滤波了
        gfted = torch.matmul(mul_L, x)  # [batch_size, 1, k, node_cnt, time_step]

        # 频域处理
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)  # [batch_size, 4, 1, node_cnt, time_step]

        # 应用图卷积权重 self.weight
        igfted = torch.matmul(gconv_input, self.weight)  # [batch_size, 4, 1, node_cnt, time_step*multi]
        igfted = torch.sum(igfted, dim=1)  # [batch_size, 1, node_cnt, time_step*multi]

        # 预测路径
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))  # [batch_size, node_cnt, time_step*multi]
        forecast = self.forecast_result(forecast_source)  # [batch_size, node_cnt, time_step]

        # 重构路径（仅在第一个堆叠块中）
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)  # [batch_size, node_cnt, time_step]
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)  # [batch_size, node_cnt, time_step]
        else:
            backcast_source = None

        return forecast, backcast_source

@register("StemGNN")
class StemGNN(STBase):
    """
    基于图卷积和时间序列分析的股票预测模型
    整合了图注意力机制、图卷积和时间序列处理
    """

    def __init__(self,
                 node_cnt: int = 400,  # 节点数量
                 time_step: int = 6,  # 时间序列长度
                 multi_layer: int = 5,  # 多层因子，用于控制输出通道的大小
                 stack_cnt: int = 2,  # StockBlockLayer 的堆叠数量，2层
                 units: int = 32,  # 隐藏单元的数量
                 horizon: int = 3,  # 预测时间步长，默认为 3
                 dropout_rate: float = 0.5,  # Dropout 比率，默认为 0.5
                 leaky_rate: float = 0.2,  # LeakyReLU 的负斜率，默认为 0.2
                 *args, **kwargs):
        """
        初始化模型

        参数:
            - units (int): 隐藏单元的数量
            - stack_cnt (int): StockBlockLayer 的堆叠数量
            - time_step (int): 时间序列长度
            - multi_layer (int): 多层因子，用于控制输出通道的大小
            - horizon (int): 预测时间步长，默认为 1
            - dropout_rate (float): Dropout 比率，默认为 0.5
            - leaky_rate (float): LeakyReLU 的负斜率，默认为 0.2
            - device (str): 运行设备，默认为 'cpu'
        """
        super(StemGNN, self).__init__(*args, **kwargs)
        self.unit = units
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.node_cnt = node_cnt
        self.seq_len = self.time_step
        self.pred_len = self.horizon

        # 图注意力机制的参数
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))  # [unit, 1]
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))  # [unit, 1]
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        # GRU 层用于处理时间序列
        self.GRU = nn.GRU(self.time_step, self.unit)  # 输入大小为 time_step，隐藏大小为 unit

        self.multi_layer = multi_layer

        # 创建 StockBlockLayer 堆叠
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)]
        )

        # 最终的预测层
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

    def get_laplacian(self, graph, normalize):
        """
        计算图的拉普拉斯矩阵

        参数:
            - graph (Tensor): 邻接矩阵，形状为 [node_cnt, node_cnt]
            - normalize (bool): 是否归一化拉普拉斯矩阵

        返回:
            - L (Tensor): 拉普拉斯矩阵，形状为 [node_cnt, node_cnt]
        """
        if normalize:
            # 归一化拉普拉斯：L = I - D^(-1/2) * A * D^(-1/2)
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))  # [node_cnt, node_cnt]
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            # 非归一化拉普拉斯：L = D - A
            D = torch.diag(torch.sum(graph, dim=-1))  # [node_cnt, node_cnt]
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        计算切比雪夫多项式

        参数:
            - laplacian (Tensor): 拉普拉斯矩阵，形状为 [node_cnt, node_cnt]

        返回:
            - multi_order_laplacian (Tensor): 多阶切比雪夫多项式，形状为 [k, node_cnt, node_cnt]
                                             k 是切比雪夫多项式的阶数 (K+1)，这里 K=3
        """
        N = laplacian.size(0)  # 节点数量
        laplacian = laplacian.unsqueeze(0)  # [1, node_cnt, node_cnt]

        # 计算不同阶数的切比雪夫多项式
        # T_0(L) = I
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        # T_1(L) = L
        second_laplacian = laplacian
        # T_2(L) = 2*L*T_1(L) - T_0(L)
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        # T_3(L) = 2*L*T_2(L) - T_1(L)
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian

        # 拼接所有阶数的切比雪夫多项式
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        # [4, node_cnt, node_cnt]

        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        """
        潜在相关层，用于学习节点之间的关系

        参数:
            - x (Tensor): 输入张量，形状为 [batch_size, node_cnt, time_step]

        返回:
            - mul_L (Tensor): 切比雪夫多项式转换后的拉普拉斯矩阵，形状为 [batch_size, k, node_cnt, node_cnt]
            - attention (Tensor): 注意力矩阵，形状为 [node_cnt, node_cnt]
        """
        # 使用 GRU 处理时间序列
        # 转置输入使其适合 GRU: [time_step, batch_size, node_cnt]
        # input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        # 转置回原始顺序: [batch_size, time_step, node_cnt]
        # input = input.permute(1, 0, 2).contiguous()
        input, _ = self.GRU(x.contiguous()) # [time_step, batch_size, node_cnt]
        # print('input shape', input.shape)

        # 计算自注意力
        attention = self.self_graph_attention(input)  # [batch_size, node_cnt, node_cnt]
        attention = torch.mean(attention, dim=0)  # [node_cnt, node_cnt]

        # 计算度矩阵
        degree = torch.sum(attention, dim=1)  # [node_cnt]

        # 确保注意力矩阵是对称的，视为邻接矩阵A
        attention = 0.5 * (attention + attention.T)

        # 计算拉普拉斯矩阵：L = D^(1/2) * (D - A) * D^(1/2)
        degree_l = torch.diag(degree)  # [node_cnt, node_cnt]
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))  # [node_cnt, node_cnt]
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # [node_cnt, node_cnt]

        # 计算切比雪夫多项式
        mul_L = self.cheb_polynomial(laplacian)  # [k, node_cnt, node_cnt]

        return mul_L, attention

    def self_graph_attention(self, input):
        """
        自注意力机制，用于学习节点之间的关系

        参数:
            - input (Tensor): 输入张量，形状为 [batch_size, time_step, unit]

        返回:
            - attention (Tensor): 注意力矩阵，形状为 [batch_size, node_cnt, node_cnt]
        """
        # 调整输入维度
        # input = input.permute(0, 2, 1).contiguous()  # [batch_size, unit, time_step]
        bat, N, fea = input.size()

        # 计算键和查询
        key = torch.matmul(input, self.weight_key)  # [batch_size, unit, 1]
        query = torch.matmul(input, self.weight_query)  # [batch_size, unit, 1]

        # 计算注意力分数
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)  # [batch_size, N*N, 1]
        data = data.squeeze(2)  # [batch_size, N*N]
        data = data.view(bat, N, -1)  # [batch_size, N, N]

        # 应用 LeakyReLU 激活函数
        data = self.leakyrelu(data)

        # 应用 softmax 归一化
        attention = F.softmax(data, dim=2)  # [batch_size, N, N]

        # 应用 dropout
        attention = self.dropout(attention)

        return attention

    def graph_fft(self, input, eigenvectors):
        """
        图傅里叶变换

        参数:
            - input (Tensor): 输入张量
            - eigenvectors (Tensor): 特征向量

        返回:
            - output (Tensor): 输出张量
        """
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        """
        前向传播

        参数:
            - x (Tensor): 输入张量，形状为 [batch_size, node_cnt, time_step]

        返回:
            - forecast (Tensor): 预测输出，形状为 [batch_size, horizon, node_cnt] 或 [batch_size, node_cnt]
            - attention (Tensor): 注意力矩阵，形状为 [node_cnt, node_cnt]
        """
        # 计算潜在相关性和切比雪夫多项式
        # attention：邻接矩阵 A，形状为 [node_cnt, node_cnt]
        # mul_L：对应A的L的4-order切比雪夫多项式，形状为 [k, node_cnt, node_cnt]
        mul_L, attention = self.latent_correlation_layer(x)  # [k, node_cnt, node_cnt], [node_cnt, node_cnt]

        # 调整输入维度以适应 StockBlockLayer
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # [batch_size, 1, time_step, node_cnt]

        # 通过堆叠的 StockBlockLayer 处理
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X,
                                                    mul_L)  # [batch_size, node_cnt, time_step], [batch_size, node_cnt, time_step]
            result.append(forecast)

        # 前两层堆叠模块的预测结果相加
        forecast = result[0] + result[1]  # [batch_size, node_cnt, time_step]

        # 最终的预测层
        forecast = self.fc(forecast)  # [batch_size, node_cnt, horizon]

        # 根据 horizon 调整输出维度
        if forecast.size()[-1] == 1:
            # 如果 horizon = 1，则输出形状为 [batch_size, node_cnt]
            return forecast.unsqueeze(1).squeeze(-1)
        else:
            # 否则，输出形状为 [batch_size, horizon, node_cnt]
            return forecast.contiguous()

if __name__ == '__main__':
    model = StemGNN(node_cnt=400, time_step = 6, horizon=3, stack_cnt=2, multi_layer=2, units=32)
    dummy_input = torch.randn(32, 400, 6)
    output, attention = model(dummy_input)
    # output.shape: torch.Size([32, 400, 3])
    # attention.shape: torch.Size([400, 400])