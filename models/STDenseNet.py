from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import L1Loss

from models.STBase import STBase


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([input, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNetUnit(nn.Sequential):
    def __init__(self, channels, nb_flows, layers=5, growth_rate=12,
                 num_init_features=32, bn_size=4, drop_rate=0.2):
        super(DenseNetUnit, self).__init__()

        if channels > 0:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, padding=1)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True))
            ]))

            # Dense Block
            num_features = num_init_features
            num_layers = layers
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock', block)
            num_features = num_features + num_layers * growth_rate

            # Final batch norm
            self.features.add_module('normlast', nn.BatchNorm2d(num_features))
            self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                           kernel_size=1, padding=0, bias=False))

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        else:
            pass

    def forward(self, x):
        out = self.features(x)
        out = out.squeeze()
        return out


class STDenseNet(STBase, LightningModule):
    def __init__(self, 
                 channels: list=[3,3,0],
                 **kwargs):
        """
        :param list channels: define channels of different Unit, a list of [close, peoriod, trend], defaults to [3,3,0]
        """
        kwargs['reduceLRPatience'] = 10
        super(STDenseNet, self).__init__(**kwargs)
        if len(channels) != 3:
            raise ValueError("The length of channels should be 3.")
        self.channels_close = channels[0]
        self.channels_period = channels[1]
        self.channels_trend = channels[2]
        self.save_hyperparameters()

        self.feature_close = DenseNetUnit(self.channels_close, 1)
        self.feature_period = DenseNetUnit(self.channels_period, 1)
        self.feature_trend = DenseNetUnit(self.channels_trend, 1)      

    def forward(self, x):
        xc = x[:, 0:self.channels_close, :, :]
        xp = x[:, self.channels_close:self.channels_close+self.channels_period, :, :]
        xt = x[:, self.channels_close+self.channels_period:self.channels_close+self.channels_period+self.channels_trend, :, :]
        out = self.feature_close(xc)
        if self.channels_period > 0:
            out += self.feature_period(xp)
        if self.channels_trend > 0:
            out += self.feature_trend(xt)
        return torch.sigmoid(out)
