import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from models.STBase import STBase


class MLP(STBase):
    def __init__(self, *, mlp_dim = 64, 
                          pred_len = 1, 
                          close_len = 3, 
                          period_len = 0,
                          **kwargs):
        super(MLP, self).__init__(**kwargs)
        seq_len = close_len + period_len
        self.pred_len = pred_len
        self.MLP = nn.Sequential(nn.Linear(seq_len, mlp_dim),
                                 nn.ReLU(),
                                 nn.Linear(mlp_dim, mlp_dim),
                                 nn.ReLU(),
                                 nn.Linear(mlp_dim, pred_len))

    def forward(self, img):
        # input.shape: [64, 64, 1, 20, 20]
        img = img.squeeze(2) # [64, 64, 20, 20]
        # print('img shape: ', img.shape)
        x = rearrange(img, 'b s h w -> b h w s')
        x = self.MLP(x)
        x = rearrange(x, 'b h w s -> b s h w')
        return x
