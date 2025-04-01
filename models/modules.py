import torch
import torch.nn as nn
import numpy as np
from torch.functional import norm
from torch.nn import init



class DGG(nn.Module):
    """
    Differentiable graph generator for ICLR
    """

    def __init__(self, in_dim=32, latent_dim=64, args=None):
        super().__init__()

        self.args = args

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU()
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(latent_dim + self.args.extra_edge_dim, latent_dim),
            nn.LeakyReLU()
        )

        # Degree estimator
        self.degree_decoder = nn.Sequential(
            nn.Linear(1, 1, bias=True),
            nn.LeakyReLU()
        )

        # Top-k selector
        self.var_grads = {"edge_p": [], "first_k": [], "out_adj": []}

    def forward(self, x, adj, noise=True, writer=None, epoch=None):
        """

        Args:
            x:
            adj:
            noise:
            writer:
            epoch:

        Returns:

        """

        assert x.ndim == 2
        assert len(adj.shape) == 2

        N = x.shape[0]  # number of nodes, 2708

        # Encode node features [N, in_dim] --> [N, h], h = latent_dim
        # before node encoder, x.shape:  torch.Size([2708, 1433])
        x = self.node_encoder(x)
        # after node encoder, x.shape:  torch.Size([2708, 64])

        # Rank edges using encoded node features [N, h] --> [E]
        # adj.sum().item() = 23549
        u = x[adj.indices()[0, :]]  # [E, h]，所有边的起点的特征向量
        v = x[adj.indices()[1, :]]  # [E, h]，所有边的终点的特征向量
        uv_diff = u - v  # 两端节点的特征差值，[E, h]

        # before edge encoder, uv_diff.shape:  torch.Size([23549, 64])
        edge_feat = self.edge_encoder(uv_diff)  # 将特征差值喂入MLP，得到边嵌入[E, h]
        # after edge encoder, edge_feat.shape:  torch.Size([23549, 64])

        edge_rank = edge_feat.sum(-1)  # [E, 1]，对特征向量求和，得到每条边的重要性分数
        edge_rank = torch.sigmoid(edge_rank)  # [E, 1]，将重要性分数映射到 [0, 1] 范围, [23549, 1]
        # 重新构造成稀疏矩阵形式
        edge_rank = torch.sparse.FloatTensor(adj.indices(), edge_rank, adj.shape)
        # 转换为密集矩阵表示 [N, N]
        edge_rank = edge_rank.to_dense()

        # Estimate node degree using encoded node features and edge rankings
        # 将边的重要性分数按行（即节点维度）求和，得到每个节点的度 k（形状为 [N, 1]）
        k = edge_rank.sum(-1).unsqueeze(-1)
        # 将边的重要性分数送入MLP，计算嵌入
        k = self.degree_decoder(k)  # [N, 1]

        # Select top-k edges
        # sort edge ranks descending
        srt_edge_rank, idxs = torch.sort(edge_rank, dim=-1, descending=True)

        t = torch.arange(N).reshape(1, N).cuda()  # base domain [1, N]
        # k = k.unsqueeze(0)                          # [N, 1]
        w = 1  # sharpness parameter
        # 通过tanh计算一个软选择掩码first_k，控制每个节点选择的边数。
        # 这种方式能够实现可微分的 top-k 操作
        first_k = 1 - 0.5 * (
            1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1
        first_k = first_k + 1.0

        # Multiply edge rank by first k
        # 对应于论文中的 e✖️h
        first_k_ranks = srt_edge_rank * first_k

        # Unsort
        out_adj = first_k_ranks.clone().scatter_(dim=-1, index=idxs, src=first_k_ranks)
        # out_adj = out_adj[adj.indices()[0, :], adj.indices()[1, :]]
        # out_adj = torch.sparse.FloatTensor(adj.indices(), out_adj, adj.shape)

        # return top-k edges and encoded node features
        return out_adj.to_sparse(), x






#############################################################################################
# 多尺度注意力模块（EMSA），用于实现多尺度注意力机制
# if __name__ == '__main__':
#     block = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True).cuda()# 创建EMSA模块实例，并配置到CUDA上（如果可用）
#     input = torch.rand(64, 64, 512).cuda()# 随机生成输入数据
#     output = block(input, input, input)# 前向传播
#     print(output.shape)
class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):

        super(EMSA, self).__init__()

        self.H =  H# 输入特征图的高度
        self.W = W  # 输入特征图的宽度
        self.fc_q = nn.Linear(d_model, h * d_k)  # 查询向量的全连接层
        self.fc_k = nn.Linear(d_model, h * d_k)  # 键向量的全连接层
        self.fc_v = nn.Linear(d_model, h * d_v)  # 值向量的全连接层
        self.fc_o = nn.Linear(h * d_v, d_model)  # 输出的全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合

        self.ratio = ratio  # 空间降采样比例
        if (self.ratio > 1):
            # 如果空间降采样比例大于1，添加空间降采样层
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2,
                                     groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            # 如果应用变换，添加变换层
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()
        # 初始化权重

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq, c = queries.shape
        nk = keys.shape[1]
        # 生成查询、键和值向量
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            # 如果空间降采样，处理查询以生成键和值向量
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            x = self.sr_conv(x)  # bs,c,h,w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            # 不进行空间降采样，直接生成键和值向量
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            # 应用变换计算注意力权重
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            # 直接计算注意力权重
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)  # 应用dropout
        # 计算输出
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out  # 返回输出结果



#############################################################################################
# 定义XNorm函数，对输入x进行规范化
def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor

# UFOAttention类继承自nn.Module
# if __name__ == '__main__':
#     # 示例用法
#     block = UFOAttention(d_model=512, d_k=512, d_v=512, h=8).cuda()
#     input = torch.rand(64, 64, 512).cuda()
#     output = block(input, input, input)
#     print(output.shape)

class UFOAttention(nn.Module):
    '''
    实现一个改进的自注意力机制，具有线性复杂度。
    '''

    # 初始化函数
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: 模型的维度
        :param d_k: 查询和键的维度
        :param d_v: 值的维度
        :param h: 注意力头数
        '''
        super(UFOAttention, self).__init__()
        # 初始化四个线性层：为查询、键、值和输出转换使用
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        # gamma参数用于规范化
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 前向传播
    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # 通过线性层将查询、键、值映射到新的空间
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # 计算键和值的乘积，然后对结果进行规范化
        kv = torch.matmul(k, v)  # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)  # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out


#############################################################################################
# 《An Attention Free Transformer》
# if __name__ == '__main__':
#     block = AFT_FULL(d_model=512, n=64).cuda()
#     input = torch.rand(64, 64, 512).cuda()
#     output = block(input)
#     print(output.shape) # 打印输出形状

class AFT_FULL(nn.Module):
    # 初始化AFT_FULL模块
    def __init__(self, d_model, n=49, simple=False):
        super(AFT_FULL, self).__init__()
        # 定义QKV三个线性变换层
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        # 根据simple参数决定位置偏置的初始化方式
        if (simple):
            self.position_biases = torch.zeros((n, n))  # 简单模式下为零矩阵
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))  # 非简单模式下为可学习的参数
        self.d_model = d_model
        self.n = n  # 输入序列的长度
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid函数

        self.init_weights()  # 初始化模型权重

    def init_weights(self):
        # 对模块中的参数进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        bs, n, dim = input.shape  # 输入的批大小、序列长度和特征维度

        # 通过QKV变换生成查询、键和值
        q = self.fc_q(input)  # bs,n,dim
        k = self.fc_k(input).view(1, bs, n, dim)  # 1,bs,n,dim，为了后续运算方便
        v = self.fc_v(input).view(1, bs, n, dim)  # 1,bs,n,dim

        # 使用位置偏置和键值对进行加权求和
        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n,bs,dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n,bs,dim

        # 计算加权求和的结果，并通过sigmoid函数调制查询向量
        out = (numerator / denominator)  # n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs,n,dim，最后将结果重新排列

        return out



# -- Bernoulli variational auto-encoder.
#
# require('math')
# require('torch')
# require('nn')
# require('cunn')
# require('optim')
# require('image')
#
# -- (1) The Kullback Leiber loss follows the Kullback Leibler loss of the Gaussian VAE.
# -- The Kullback-Leibler divergence between two Bernoulli distribution can easily
# -- be written down by summing over all possible states (i.e. 0 and 1).
# --- @class KullbackLeiberDivergence
# local KullbackLeiberDivergence, KullbackLeiberDivergenceParent = torch.class('nn.KullbackLeiberDivergence', 'nn.Module')
#
# --- Initialize.
# -- @param lambda weight of loss
# function KullbackLeiberDivergence:__init(lambda, sizeAverage)
#   self.lambda = lambda or 1
#   self.prior = 0.5
#   self.sizeAverage = sizeAverage or false
#   self.loss = 0
# end
#
# --- Compute the Kullback-Leiber divergence; however, the input remains
# -- unchanged - the divergence is saved in KullBackLeiblerDivergence.loss.
# -- @param input probabilities
# -- @return probabilities
# function KullbackLeiberDivergence:updateOutput(input)
#
#   -- (1.1) Forward pass of the KL divergence which is essentially
#   -- an expectation over the log of the quotient of two Bernoulli distributions.
#   -- Thus, considering all possible states (0, 1), this can be computed directly.
#   self.loss = torch.cmul(input, torch.log(input + 1e-20) - torch.log(self.prior))
#     + torch.cmul(1 - input, torch.log(1 - input + 1e-20) - torch.log(1 - self.prior))
#   self.loss = self.lambda*torch.sum(self.loss)
#
#   if self.sizeAverage then
#     self.loss = self.loss/lib.utils.storageProd(#input)
#   end
#
#   self.output = input
#   return self.output
# end
#
# --- Compute the backward pass of the Kullback-Leibler Divergence.
# -- @param input probabilities
# -- @param gradOutput gradients from top layer
# -- @return gradients from top layer plus gradient of KL divergence with respect to probabilities
# function KullbackLeiberDivergence:updateGradInput(input, gradOutput)
#
#   -- (1.2) Backward pass, i.e. derivative of (1.1).
#   local ones = input:clone():fill(1)
#   self.gradInput = torch.log(input + 1e-20) + 1 - torch.log(self.prior) - torch.cdiv(ones, 1 - input + 1e-20)
#     - torch.log(1 - input + 1e-20) + torch.cdiv(input, 1 - input + 1e-20) + torch.log(1 - self.prior)
#   self.gradInput = self.lambda*self.gradInput
#   --assert(not torch.any(self.gradInput:ne(self.gradInput)))
#
#   if self.sizeAverage then
#     self.gradInput = self.gradInput/lib.utils.storageProd(#input)
#   end
#
#   self.gradInput = self.gradInput + gradOutput
#   --print(torch.mean(self.gradInput))
#   return self.gradInput
# end
#
# -- (2) The reparameterization trick assumes that the next layer is a Sigmoid layer
# -- in order to function correctly.
# --- @class ReparameterizationSampler
# local ReparameterizationSampler, ReparameterizationSamplerParent = torch.class('nn.ReparameterizationSampler', 'nn.Module')
#
# --- Initialize.
# -- @param temperature temperature of prediction
# function ReparameterizationSampler:__init(temperature)
#   self.temperature = temperature or 1
# end
#
# --- Sample from the provided mean and variance using the reparameterization trick.
# -- @param input Bernoulli probabilities
# -- @return sample
# function ReparameterizationSampler:updateOutput(input)
#
#   -- (2.1) Reparameterization:
#   -- Let u be a uniform random variale in [0,1], p be the predicted probability (i.e. input),
#   -- let l be the temperature.
#   -- y = sigmoid((log(p) + log(u) - log(1 - u))/l)
#   self.eps = torch.rand(input:size()):cuda()
#
#   --self.output = (torch.log(input + 1e-20) + torch.log(self.eps) - torch.log(1 - self.eps))/self.temperature
#   self.output = (torch.log(input + 1e-20) - torch.log(-torch.log(self.eps + 1e-20) + 1e-20))/self.temperature
#   --print(torch.sigmoid(self.output))
#   return self.output
# end
#
# --- Backward pass of the sampler.
# -- @param input Bernoulli probabilities
# -- @param gradOutput gradients of top layer
# -- @return gradients with respect to input, table of two elements
# function ReparameterizationSampler:updateGradInput(input, gradOutput)
#
#   -- (2.2) Derivative of reparameterization with respect to p.
#   --local ones = input:clone():fill(1)
#   --self.gradInput = torch.cmul(torch.cdiv(ones, input*self.temperature + 1e-20), gradOutput)
#   self.gradInput = torch.cdiv(gradOutput, input + 1e-20)/self.temperature
#   --assert(not torch.any(self.gradInput:ne(self.gradInput)))
#   --print(torch.mean(self.gradInput))
#   return self.gradInput
# end
#
# -- Data parameters.
# H = 24
# W = 24
# rH = 8
# rW = 8
# N = 50000
#
# -- Fix random seed.
# torch.manualSeed(1)
#
# inputs = torch.Tensor(N, 1, H, W):fill(0)
# for i = 1, N do
#   local h = torch.random(rH, rH)
#   local w = torch.random(rW, rW)
#   local aH = torch.random(1, H - h)
#   local aW = torch.random(1, W - w)
#   inputs[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
# end
#
# outputs = inputs:clone()
#
# -- (3) The encoder consists of several linear layerReparameterizationSamplers followed by
# -- the Kullback Leibler loss, the samples and the docoder; the decoder
# -- mirrors the encoder.
# -- (3.1) The encoder, as for vanilla VAE.
# hidden = math.floor(2*H*W)
# encoder = nn.Sequential()
# encoder:add(nn.View(1*H*W))
# encoder:add(nn.Linear(1*H*W, hidden))
# --encoder:add(nn.BatchNormalization(hidden))
# encoder:add(nn.ReLU(true))
# encoder:add(nn.Linear(hidden, hidden))
# --encoder:add(nn.BatchNormalization(hidden))
# encoder:add(nn.ReLU(true))
#
# code = 25
# encoder:add(nn.Linear(hidden, code))
#
# -- (3.2) As for vanilla VAEs.
# decoder = nn.Sequential()
# decoder:add(nn.Linear(code, hidden))
# --decoder:add(nn.BatchNormalization(hidden))
# decoder:add(nn.ReLU(true))
# decoder:add(nn.Linear(hidden, hidden))
# --decoder:add(nn.BatchNormalization(hidden))
# decoder:add(nn.ReLU(true))
# decoder:add(nn.Linear(hidden, 1*H*W))
# decoder:add(nn.View(1, H, W))
# decoder:add(nn.Sigmoid(true))
#
# -- (3) The full model, i.e encoder followed by the Kullback Leibler
# -- divergence and the reparameterization trick sampler.
# -- The main difference to the Gaussian model is that a Sigmoid layer follows
# -- the reparameterization sampler.
# model = nn.Sequential()
# model:add(encoder)
# KLD = nn.KullbackLeiberDivergence()
# model:add(nn.Sigmoid(true))
# model:add(KLD)
# model:add(nn.ReparameterizationSampler())
# model:add(nn.Sigmoid(true))
# model:add(decoder)
# print(model)
# model = model:cuda()
#
# criterion = nn.BCECriterion()
# criterion.sizeAverage = false
# criterion = criterion:cuda()
#
# parameters, gradParameters = model:getParameters()
# parameters = parameters:cuda()
# gradParameters = gradParameters:cuda()
#
# batchSize = 16
# learningRate = 0.001
# epochs = 10
# iterations = epochs*math.floor(N/batchSize)
# lossIterations = 50 -- in which interval to report training
# protocol = torch.Tensor(iterations, 2):fill(0)
#
# for t = 1, iterations do
#
#   -- Sample a random batch from the dataset.
#   local shuffle = torch.randperm(N)
#   shuffle = shuffle:narrow(1, 1, batchSize)
#   shuffle = shuffle:long()
#
#   local input = inputs:index(1, shuffle)
#   local output = outputs:index(1, shuffle)
#
#   input = input:cuda()
#   output = output:cuda()
#
#   --- Definition of the objective on the current mini-batch.
#   -- This will be the objective fed to the optimization algorithm.
#   -- @param x input parameters
#   -- @return object value, gradients
#   local feval = function(x)
#
#     -- Get new parameters.
#     if x ~= parameters then
#       parameters:copy(x)
#     end
#
#     -- Reset gradients.
#     gradParameters:zero()
#
#     -- Evaluate function on mini-batch.
#     local pred = model:forward(input)
#     local f = criterion:forward(pred, output)
#
#     protocol[t][1] = f
#     protocol[t][2] = KLD.loss
#
#     -- Estimate df/dW.
#     local df_do = criterion:backward(pred, output)
#     model:backward(input, df_do)
#
#     -- return f and df/dX
#     return f, gradParameters
#   end
#
#   adamState = adamState or {
#       learningRate = learningRate,
#       momentum = 0,
#       learningRateDecay = 5e-7
#   }
#
#   -- Returns the new parameters and the objective evaluated
#   -- before the update.
#   p, f = optim.adam(feval, parameters, adamState)
#
#   if t%lossIterations == 0 then
#     local loss = torch.mean(protocol:narrow(2, 1, 1):narrow(1, t - lossIterations + 1, lossIterations))
#     local KLDLoss = torch.mean(protocol:narrow(2, 2, 1):narrow(1, t - lossIterations + 1, lossIterations))
#     print('[Training] ' .. t .. '/' .. iterations .. ': ' .. loss .. ' | ' .. KLDLoss)
#   end
# end
#
# randoms = torch.Tensor(20 * H, 20 * W)
#
# -- Sample 20 x 20 points
# for i = 1, 20  do
#   for j = 1, 20 do
#     local sample = torch.rand(1, code)
#     sample[sample:gt(0.5)] = 1
#     sample[sample:lt(1)] = 0
#     local random = decoder:forward(sample:cuda())
#     random = random:float()
#     randoms[{{(i - 1) * H + 1, i * H}, {(j - 1) * W + 1, j * W}}] = random
#   end
# end
#
# image.save('random.png', randoms)