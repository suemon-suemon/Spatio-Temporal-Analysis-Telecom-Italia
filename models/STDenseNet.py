from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.STBase import STBase


class _DenseLayer(nn.Sequential):
    """
    DenseNet的基本层结构，实现了特征的密集连接
    输入维度: [batch_size, num_input_features, height, width]
    输出维度: [batch_size, num_input_features + growth_rate, height, width]
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        参数:
            num_input_features: 输入特征通道数
            growth_rate: 每层新增特征通道数
            bn_size: 用于压缩模型的瓶颈因子
            drop_rate: Dropout概率
        """
        super(_DenseLayer, self).__init__()
        # BN-ReLU-Conv1x1结构(特征通道压缩)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))  # 输入输出维度不变，仅归一化
        self.add_module('relu1', nn.ReLU(inplace=True))  # 输入输出维度不变，仅激活
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1,
                                           bias=False))  # 输出维度: [batch_size, bn_size*growth_rate, height, width]

        # BN-ReLU-Conv3x3结构(特征提取)
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))  # 输入输出维度不变，仅归一化
        self.add_module('relu2', nn.ReLU(inplace=True))  # 输入输出维度不变，仅激活
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))  # 输出维度: [batch_size, growth_rate, height, width]
        self.drop_rate = drop_rate

    def forward(self, input):
        """
        前向传播：处理输入特征并与原始输入级联
        输入维度: [batch_size, num_input_features, height, width]
        输出维度: [batch_size, num_input_features + growth_rate, height, width]
        """
        new_features = super(_DenseLayer, self).forward(input)  # 维度: [batch_size, growth_rate, height, width]
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)  # 维度不变
        # 核心DenseNet操作：将新特征与输入特征进行级联
        return torch.cat([input, new_features], 1)  # 维度: [batch_size, num_input_features + growth_rate, height, width]


class _DenseBlock(nn.Sequential):
    """
    DenseBlock：由多个_DenseLayer组成
    输入维度: [batch_size, num_input_features, height, width]
    输出维度: [batch_size, num_input_features + num_layers*growth_rate, height, width]
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        """
        参数:
            num_layers: 块内_DenseLayer的数量
            num_input_features: 输入特征通道数
            bn_size: 瓶颈因子
            growth_rate: 每层新增特征通道数
            drop_rate: Dropout概率
        """
        super(_DenseBlock, self).__init__()
        # 逐层构建，每层输入通道数递增
        for i in range(num_layers):
            # 第i层的输入通道数 = 初始通道数 + 前i层累积的新增通道数
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            # 每层输出通道增加growth_rate，累积效应


class DenseNetUnit(nn.Sequential):
    """
    DenseNet单元：基于输入通道数创建一个完整的DenseNet结构单元
    输入维度: [batch_size, channels, height, width]
    输出维度: [batch_size, nb_flows, height, width]
    """

    def __init__(self, channels, nb_flows, layers=5, growth_rate=12,
                 num_init_features=32, bn_size=4, drop_rate=0.2):
        """
        参数:
            channels: 输入通道数，如果为0则不执行任何操作
            nb_flows: 输出通道数(流的数量)
            layers: Dense块中的层数
            growth_rate: 每层新增特征通道数
            num_init_features: 初始卷积层输出通道数
            bn_size: 瓶颈因子
            drop_rate: Dropout概率
        """
        super(DenseNetUnit, self).__init__()

        if channels > 0:
            # 初始卷积处理
            self.features = nn.Sequential(OrderedDict([
                # 输入维度: [batch_size, channels, height, width]
                # 输出维度: [batch_size, num_init_features, height, width]
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, padding=1)),
                ('norm0', nn.BatchNorm2d(num_init_features)),  # 维度不变
                ('relu0', nn.ReLU(inplace=True))  # 维度不变
            ]))

            # 创建Dense块
            num_features = num_init_features  # 当前特征通道数
            num_layers = layers  # 块内层数
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # 输入维度: [batch_size, num_init_features, height, width]
            # 输出维度: [batch_size, num_init_features + num_layers*growth_rate, height, width]
            self.features.add_module('denseblock', block)
            num_features = num_features + num_layers * growth_rate  # 更新通道数

            # 最终处理：批量归一化和1x1卷积调整通道数
            self.features.add_module('normlast', nn.BatchNorm2d(num_features))  # 维度不变
            # 输入维度: [batch_size, num_features, height, width]
            # 输出维度: [batch_size, nb_flows, height, width]
            self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                           kernel_size=1, padding=0, bias=False))

            # 权重初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)  # He初始化
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        else:
            # 当channels为0时，不执行任何操作
            pass

    def forward(self, x):
        """
        前向传播
        输入维度: [batch_size, channels, height, width]
        输出维度: [batch_size, nb_flows, height, width]
        """
        out = self.features(x)
        return out


class STDenseNet(STBase):
    """
    时空DenseNet模型：基于DenseNet的时空预测模型
    整合了近期、周期和趋势特征
    """

    def __init__(self,
                 channels: list = [3, 0, 0],  # 默认只使用近期通道
                 pred_len: int = 3,
                 **kwargs):
        """
        初始化STDenseNet模型
        参数:
            channels: 定义不同单元的通道，格式为[close, period, trend]
                - close: 近期特征通道数
                - period: 周期特征通道数
                - trend: 趋势特征通道数
            pred_len: 预测长度
        """
        kwargs['reduceLRPatience'] = 20  # 设置学习率降低的耐心值
        super(STDenseNet, self).__init__(**kwargs)
        if len(channels) != 3:
            raise ValueError("The length of channels should be 3.")
        self.channels_close = channels[0]  # 近期特征通道数
        self.channels_period = channels[1]  # 周期特征通道数
        self.channels_trend = channels[2]  # 趋势特征通道数
        self.seq_len = self.channels_close  # 序列长度等于近期特征通道数
        self.pred_len = pred_len  # 预测长度
        self.save_hyperparameters()  # 保存超参数

        # 创建不同特征的DenseNet单元
        self.close_feature = DenseNetUnit(self.channels_close, self.pred_len)  # 近期特征单元
        if self.channels_period > 0:
            self.period_feature = DenseNetUnit(self.channels_period, self.pred_len)  # 周期特征单元
        if self.channels_trend > 0:
            self.trend_feature = DenseNetUnit(self.channels_trend, self.pred_len)  # 趋势特征单元

    def forward(self, x):
        """
        前向传播
        输入维度: [batch_size, total_channels, height, width]，其中total_channels = channels_close + channels_period + channels_trend
        输出维度: [batch_size, 1, height, width]
        """
        x = x.squeeze()  # 移除维度为1的维度

        # 将输入按不同特征类型分割
        # 近期特征 - 维度: [batch_size, channels_close, height, width]
        xc = x[:, 0:self.channels_close, :, :]

        # 周期特征 - 维度: [batch_size, channels_period, height, width]
        xp = x[:, self.channels_close:self.channels_close + self.channels_period, :, :]

        # 趋势特征 - 维度: [batch_size, channels_trend, height, width]
        xt = x[:,
             self.channels_close + self.channels_period:self.channels_close + self.channels_period + self.channels_trend,
             :, :]

        out = 0  # 初始化输出

        # 融合不同特征的预测结果
        if self.channels_close > 0:
            out += self.close_feature(xc)  # 近期特征的贡献
        if self.channels_period > 0:
            out += self.period_feature(xp)  # 周期特征的贡献
        if self.channels_trend > 0:
            out += self.trend_feature(xt)  # 趋势特征的贡献

        out = torch.sigmoid(out)
        # 使用sigmoid函数归一化输出到(0,1)区间
        # 输出维度: [batch_size, 1, height, width]
        return out

if __name__ == '__main__':
    # 创建一个STDenseNet实例
    model = STDenseNet(channels=[3, 0, 0], pred_len=3)

    # 创建一个输入张量
    input_tensor = torch.randn(32, 6, 1, 20, 20)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)