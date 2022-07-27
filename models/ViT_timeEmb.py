import math
from collections import OrderedDict

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from models.dcn import DeformableConv2d
from models.informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.informer.attn import FullAttention, ProbAttention, AttentionLayer
from models.STBase import STBase
from models.convnext import ConvNeXt, LayerNorm


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MLPDecoder(nn.Module):
    def __init__(self, pred_len, n_input, stride_size, img_size, *, n_hidden, c_groups, inner_channels=16, grid_size=30):
        super(MLPDecoder, self).__init__()
        self.grid_height, self.grid_width = pair(grid_size)
        self.c_groups = c_groups
        self.patch_height, self.patch_width = pair(stride_size)
        self.img_height, self.img_width = pair(img_size)
        # if self.patch_height > 1:
        #     temporal_in = pred_len * inner_channels
        # else:
        #     temporal_in = n_input * c_groups
         
        
        # self.mlp_temporal = nn.Sequential(nn.Conv2d(temporal_in, n_hidden, kernel_size=1, padding=0),
        #                                  nn.GELU(),
        #                                  nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
        #                                  nn.GELU(),
        #                                  nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
        #                                  nn.GELU(),
        #                                  nn.Conv2d(n_hidden, pred_len, kernel_size=1, padding=0))

    def forward(self, x):

        # x = self.mlp_temporal(x)
        return x


class ViT_timeEmb(STBase):
    def __init__(self, *, image_size=(30, 30), # (11, 11)
                          patch_size, # (3, 3)
                          stride_size, # (2, 2)
                          padding_size,
                          dim, 
                          depth, 
                          heads, 
                          mlp_dim, 
                          pred_len = 1, 
                          pool = 'cls', 
                          close_len = 3, 
                          period_len = 0,
                          channels_group = 8,
                          conv_channels = 16,
                          dccnn_layers = 5,
                          dccnn_growth_rate = 64,
                          dccnn_init_channels = 64,
                          inner_channels = 16,
                          d_decoder = 256,
                          dropout = 0.1, 
                          **kwargs):
        super(ViT_timeEmb, self).__init__(**kwargs)
        self.seq_len = close_len + period_len * pred_len
        self.close_len = close_len
        self.period_len = period_len
        self.pred_len = pred_len
        self.channels_group = channels_group
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        stride_height, stride_width = pair(stride_size)
        padding_height, padding_width = pair(padding_size)
        self.patch_height = patch_height
        self.patch_width = patch_width

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (((image_height - patch_height) // stride_height + 2 * padding_height + 1) * 
                      ((image_width - patch_width) // stride_width + 2 * padding_width + 1)) * channels_group
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_features = nn.Sequential(
            nn.Conv2d(self.seq_len, conv_channels*self.seq_len, kernel_size=3, padding='same', groups=self.seq_len),
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = stride_size, padding=padding_size), # (N, C, *) -> (N, C × ∏(kernel_size), L)
            Rearrange('b (cg c s) l -> b (l cg) (c s)', cg = channels_group, s = patch_dim),
            nn.Linear(patch_dim*conv_channels*self.seq_len//channels_group, dim),
            # nn.GELU(),
            # nn.Linear(d_decoder, dim)
        )

        self.pos_embedding = PositionalEncoding(dim, dropout=dropout)
        self.time_embedding = TemporalEmbedding(dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True, activation='gelu')
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, depth)
        # self.transformerEncoder = Transformer(dim, depth, heads, 64, mlp_dim, dropout)

        self.spatial = nn.Sequential(nn.ConvTranspose2d(dim*channels_group, inner_channels, kernel_size=stride_size, stride=stride_size),)

        self.mlp_temporal = nn.Sequential(nn.Conv2d(inner_channels, d_decoder, kernel_size=1, padding=0),
                                        nn.GELU(),
                                        nn.Conv2d(d_decoder, d_decoder, kernel_size=1, padding=0),
                                        nn.GELU(),
                                        nn.Conv2d(d_decoder, pred_len, kernel_size=1, padding=0))


    def forward(self, x, x_mark):
        x = x.squeeze(2)
        b, s, h, w = x.shape
        res = repeat(torch.mean(x[:, -1:], dim=1), 'b h w -> b s h w', s=self.pred_len)
        
        conv = self.conv_features(x)
        src = self.to_patch_embedding(conv)
        src = self.pos_embedding(src)

        time_embedding = self.time_embedding(x_mark)
        time_embedding = rearrange(time_embedding, 'b (g s) w -> b g s w', g=self.channels_group).mean(dim=2)
        time_embedding = repeat(time_embedding, 'b s w -> b h s w', h=self.num_patches//self.channels_group)
        time_embedding = rearrange(time_embedding, 'b h s w -> b (h s) w')
        src += time_embedding

        tr_en = self.transformerEncoder(src)

        x = rearrange(tr_en, 'b (h w cg) d -> b (cg d) h w', 
                         cg=self.channels_group, 
                         h=h//self.patch_height, 
                         w=w//self.patch_width)
        if self.patch_height > 1:
           x = self.spatial(x)

        x = self.mlp_temporal(x)
        return (x+res).unsqueeze(2)
    
    def _process_one_batch(self, batch):
        x, y, x_mark, x_meta = batch
        y_hat = self(x, x_mark)
        # if self.pred_len == 1:
        #     y = y.unsqueeze(1)
        return y_hat, y
