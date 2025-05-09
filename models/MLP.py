import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from models.STBase import STBase
from utils.registry import register

@register("MLP")
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

    def forward(self, x):

        # format: default / completion
        # batch X shape: [b, close_len, 1, n_row, n_col]
        # batch Y shape: [b, pred_len, 1, n_row, n_col]

        if isinstance(x, list):
            x = x[0]  # mask = x[1]

        x = rearrange(x, 'b t c h w -> b c h w t')
        x = self.MLP(x)
        x = rearrange(x, 'b c h w t -> b t c h w')

        return x
