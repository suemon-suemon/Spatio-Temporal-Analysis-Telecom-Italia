from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.STBase import STBase
from utils.registry import register

@register("DeDenseNet")
class _DenseLayer(nn.Sequential):
    """
    DenseNet的基本层，包含批归一化、ReLU激活和卷积操作
    实现了论文中提到的"BN-ReLU-Conv"模式，并支持dropout
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        初始化DenseLayer

        参数:
            num_input_features: 输入特征通道数
            growth_rate: 每层增长的特征通道数
            bn_size: 瓶颈层的放大系数
            drop_rate: dropout概率
        """
        super(_DenseLayer, self).__init__()
        # 第一部分：批归一化 -> ReLU -> 1x1卷积(降维)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))  # [B, num_input_features, H, W]
        self.add_module('relu1', nn.ReLU(inplace=True))  # [B, num_input_features, H, W]
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))  # [B, bn_size*growth_rate, H, W]

        # 第二部分：批归一化 -> ReLU -> 3x3卷积
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))  # [B, bn_size*growth_rate, H, W]
        self.add_module('relu2', nn.ReLU(inplace=True))  # [B, bn_size*growth_rate, H, W]
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))  # [B, growth_rate, H, W]
        self.drop_rate = drop_rate

    def forward(self, input):
        """
        前向传播：获取新特征并与输入连接

        输入:
            input: 张量，形状为 [B, num_input_features, H, W]

        返回:
            连接后的张量，形状为 [B, num_input_features + growth_rate, H, W]
        """
        new_features = super(_DenseLayer, self).forward(input)  # [B, growth_rate, H, W]
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)  # [B, growth_rate, H, W]
        # 沿通道维度连接输入和新特征，实现密集连接
        return torch.cat([input, new_features], 1)  # [B, num_input_features + growth_rate, H, W]


class _DenseBlock(nn.Sequential):
    """
    DenseBlock由多个DenseLayer组成，每个DenseLayer的输出通道数都会增加growth_rate
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        """
        初始化DenseBlock

        参数:
            num_layers: 块中DenseLayer的数量
            num_input_features: 输入特征通道数
            bn_size: 瓶颈层的放大系数
            growth_rate: 每层增长的特征通道数
            drop_rate: dropout概率
        """
        super(_DenseBlock, self).__init__()
        # 顺序添加多个DenseLayer，每层的输入通道数依次增加
        for i in range(num_layers):
            # 每个DenseLayer的输入通道数=初始通道数+之前层产生的通道数
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            # 第i层输入: [B, num_input_features + i*growth_rate, H, W]
            # 第i层输出: [B, num_input_features + (i+1)*growth_rate, H, W]


class DenseNetUnit(nn.Sequential):
    """
    完整的DenseNet单元，包含初始卷积层、DenseBlock和最终输出层
    """

    def __init__(self, channels, nb_flows, layers=5, growth_rate=12,
                 num_init_features=32, bn_size=4, drop_rate=0.2):
        """
        初始化DenseNetUnit

        参数:
            channels: 输入通道数
            nb_flows: 输出通道数/预测长度
            layers: DenseBlock中的层数
            growth_rate: 每层增长的特征通道数
            num_init_features: 初始卷积层输出的特征通道数
            bn_size: 瓶颈层的放大系数
            drop_rate: dropout概率
        """
        super(DenseNetUnit, self).__init__()

        if channels > 0:
            # 初始卷积层
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, padding=1)),
                # [B, channels, H, W] -> [B, num_init_features, H, W]
                ('norm0', nn.BatchNorm2d(num_init_features)),  # [B, num_init_features, H, W]
                ('relu0', nn.ReLU(inplace=True))  # [B, num_init_features, H, W]
            ]))

            # DenseBlock部分
            num_features = num_init_features  # 当前特征通道数
            num_layers = layers  # DenseBlock中的层数
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock', block)
            # DenseBlock输入: [B, num_init_features, H, W]
            # DenseBlock输出: [B, num_init_features + num_layers*growth_rate, H, W]

            # 更新当前特征通道数
            num_features = num_features + num_layers * growth_rate

            # 最终处理层：批归一化 + 1x1卷积 + ReLU + 批归一化 + 1x1卷积
            dim_inner = 256  # 中间层维度
            self.features.add_module('normlast', nn.BatchNorm2d(num_features))  # [B, num_features, H, W]
            self.features.add_module('convlast', nn.Conv2d(num_features, dim_inner, kernel_size=1, padding=0,
                                                           bias=False))  # [B, dim_inner, H, W]
            self.features.add_module('relulast', nn.ReLU(inplace=True))  # [B, dim_inner, H, W]
            self.features.add_module('normlast2', nn.BatchNorm2d(dim_inner))  # [B, dim_inner, H, W]
            self.features.add_module('convlast2', nn.Conv2d(dim_inner, nb_flows, kernel_size=1, padding=0,
                                                            bias=False))  # [B, nb_flows, H, W]

            # 初始化权重
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        else:
            pass  # 如果channels为0，不创建任何层

    def forward(self, x):
        """
        前向传播

        输入:
            x: 张量，形状为 [B, channels, H, W]

        返回:
            output: 张量，形状为 [B, nb_flows, H, W]
        """
        out = self.features(x)  # [B, nb_flows, H, W]
        return out


class moving_avg(nn.Module):
    """
    移动平均模块，用于突出时间序列的趋势
    """

    def __init__(self, kernel_size, stride):
        """
        初始化移动平均模块

        参数:
            kernel_size: 平均池化的核大小
            stride: 步长
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播：计算时间序列的移动平均

        输入:
            x: 张量，形状为 [B, T, C]，其中T是时间长度，C是特征维度

        返回:
            移动平均后的张量，形状为 [B, T, C]
        """
        # 对时间序列两端进行填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # [B, (kernel_size-1)//2, C]
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # [B, (kernel_size-1)//2, C]
        x = torch.cat([front, x, end], dim=1)  # [B, T+(kernel_size-1), C]

        # 转置并应用平均池化
        x = self.avg(x.permute(0, 2, 1))  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]
        return x


class series_decomp(nn.Module):
    """
    时间序列分解模块，将时间序列分解为趋势项和季节项
    """

    def __init__(self, kernel_size):
        """
        初始化时间序列分解模块

        参数:
            kernel_size: 移动平均的核大小
        """
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        前向传播：将时间序列分解为趋势项和残差(季节)项

        输入:
            x: 张量，形状为 [B, T, C]

        返回:
            res: 残差(季节)项，形状为 [B, T, C]
            moving_mean: 趋势项，形状为 [B, T, C]
        """
        # 计算移动平均作为趋势项
        moving_mean = self.moving_avg(x)  # [B, T, C]
        # 原始序列减去趋势项得到残差(季节)项
        res = x - moving_mean  # [B, T, C]
        return res, moving_mean


class DeDenseNet(STBase):
    """
    分解密集网络(DeDenseNet)模型，结合了时间序列分解和DenseNet
    用于时空预测任务
    """

    def __init__(self,
                 channels: int = 12,
                 pred_len: int = 1,

                 layers_s=4,
                 growth_rate_s=32,
                 num_init_features_s=32,
                 bn_size_s=4,

                 layers_t=4,
                 growth_rate_t=32,
                 num_init_features_t=32,
                 bn_size_t=4,

                 drop_rate=0.3,
                 kernel_size=25,
                 *args, **kwargs):
        """
        初始化DeDenseNet模型

        参数:
            channels: 输入通道数/输入序列长度
            pred_len: 预测长度
            layers_s: 季节项DenseNet的层数
            growth_rate_s: 季节项DenseNet每层增长的特征通道数
            num_init_features_s: 季节项DenseNet初始卷积层输出的特征通道数
            bn_size_s: 季节项DenseNet的瓶颈层放大系数
            layers_t: 趋势项DenseNet的层数
            growth_rate_t: 趋势项DenseNet每层增长的特征通道数
            num_init_features_t: 趋势项DenseNet初始卷积层输出的特征通道数
            bn_size_t: 趋势项DenseNet的瓶颈层放大系数
            drop_rate: dropout概率
            kernel_size: 时间序列分解的核大小
        """
        super(DeDenseNet, self).__init__(*args, **kwargs)
        self.channels_close = channels  # 输入通道数
        self.seq_len = self.channels_close  # 输入序列长度
        self.pred_len = pred_len  # 预测长度
        self.save_hyperparameters()

        # 时间序列分解模块
        self.decompsition = series_decomp(kernel_size)

        # 季节项的DenseNet
        self.seasonal = DenseNetUnit(self.channels_close, self.pred_len, layers_s,
                                     growth_rate_s, num_init_features_s, bn_size_s, drop_rate)

        # 趋势项的DenseNet
        self.trend = DenseNetUnit(self.channels_close, self.pred_len, layers_t,
                                  growth_rate_t, num_init_features_t, bn_size_t, drop_rate)

    def forward(self, x):
        """
        前向传播

        输入:
            x: 张量，形状为 [batch_size, close_len, services, n_grid_row, n_grid_col]
               其中close_len等于channels_close，表示输入序列长度
               services通常为1，表示服务类型
               n_grid_row和n_grid_col表示空间网格的维度

        返回:
            out: 预测结果，形状为 [batch_size, pred_len, services, n_grid_row, n_grid_col]
        """
        # 移除services维度(如果为1)
        x = x.squeeze(2)  # [batch_size, close_len, n_grid_row, n_grid_col]

        # 重塑并分解时间序列
        seasonal_init, trend_init = self.decompsition(x.view(x.size(0), x.size(1), -1))
        # seasonal_init: [batch_size, close_len, n_grid_row*n_grid_col]
        # trend_init: [batch_size, close_len, n_grid_row*n_grid_col]

        # 还原原始形状
        seasonal_init = seasonal_init.view(x.shape)  # [batch_size, close_len, n_grid_row, n_grid_col]
        trend_init = trend_init.view(x.shape)  # [batch_size, close_len, n_grid_row, n_grid_col]

        # 分别处理季节项和趋势项，然后相加
        out = 0
        out += self.seasonal(seasonal_init)  # [batch_size, pred_len, n_grid_row, n_grid_col]
        out += self.trend(trend_init)  # [batch_size, pred_len, n_grid_row, n_grid_col]

        # 添加services维度
        out = out.unsqueeze(2)  # [batch_size, pred_len, services=1, n_grid_row, n_grid_col]

        return out


if __name__ == '__main__':
    # 测试代码
    model = DeDenseNet(channels=6, pred_len=3)
    dummy_input = torch.randn(32, 6, 1, 20,
                              20)  # [batch_size=32, close_len=6, services=1, n_grid_row=20, n_grid_col=20]
    out = model(dummy_input)
    print(out.shape)  # 预期输出: torch.Size([32, 3, 1, 20, 20])