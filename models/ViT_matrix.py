import math

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from models.STBase import STBase


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
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
        return self.fn(self.norm(x), **kwargs)


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


class TransformerEncoder(nn.Module):
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
    def __init__(self, pred_len, n_input, patch_size, *, n_hidden, c_groups, inner_channels=16, grid_size=30):
        super(MLPDecoder, self).__init__()
        self.grid_height, self.grid_width = pair(grid_size)
        self.c_groups = c_groups
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        out_dim_s = pred_len * inner_channels * patch_height * patch_width
        self.mlp_spatial = nn.Sequential(nn.Conv2d(n_input*c_groups, n_hidden, kernel_size=1, padding=0),
                                         nn.GELU(),
                                         nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
                                         nn.GELU(),
                                         nn.Conv2d(n_hidden, out_dim_s, kernel_size=1, padding=0))
        self.mlp_temporal = nn.Sequential(nn.Conv2d(pred_len*inner_channels, n_hidden, kernel_size=1, padding=0),
                                         nn.GELU(),
                                         nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
                                         nn.GELU(),
                                         nn.Conv2d(n_hidden, pred_len, kernel_size=1, padding=0))
                                         
    def forward(self, x):
        x = rearrange(x, 'b (h w cg) d -> b (cg d) h w', 
                         cg=self.c_groups, 
                         h=30//self.patch_height, 
                         w=30//self.patch_width)
        x = self.mlp_spatial(x)
        x = rearrange(x, 'b (l dh dw) h w  -> b l (h dh) (w dw)', # l = pred_len * inner_channels
                        dh=self.patch_height,
                        dw=self.patch_width)
        x = self.mlp_temporal(x)
        return x


class TransDec(nn.Module):
    def __init__(self):
        super(TransDec, self).__init__()
    
    def forward(self, x):
        return x


class ViT_matrix(STBase):
    def __init__(self, *, image_size, # (11, 11)
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
                          channels_group = 3,
                          dropout = 0., 
                          emb_dropout = 0.,
                          **kwargs):
        super(ViT_matrix, self).__init__(**kwargs)
        self.seq_len = close_len + period_len * pred_len
        self.close_len = close_len
        self.period_len = period_len
        self.pred_len = pred_len
        self.channels_group = channels_group
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        stride_height, stride_width = pair(stride_size)
        padding_height, padding_width = pair(padding_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (((image_height - patch_height) // stride_height + 2 * padding_height + 1) * 
                      ((image_width - patch_width) // stride_width + 2 * padding_width + 1)) * channels_group
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        conv_channels = 96
        self.conv_features = nn.Sequential(
            nn.Conv2d(self.seq_len, conv_channels // 2, kernel_size=5, padding='same'),
            nn.Conv2d(conv_channels // 2, conv_channels, kernel_size=5, padding='same'),
        )

        # conv3d_channels = 48
        # self.conv3d_features = nn.Sequential(
        #     Rearrange('b t x y -> b 1 t x y'),
        #     nn.Conv3d(1, conv3d_channels // 2, kernel_size=(6, 5, 5), stride=(3, 1, 1), padding=(0, 2, 2)),
        #     nn.Conv3d(conv3d_channels // 2, conv3d_channels, kernel_size=(3, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
        #     Rearrange('b c t x y -> b (t c) x y'),
        # )

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = stride_size, padding=padding_size), # (N, C, *) -> (N, C × ∏(kernel_size), L)
            Rearrange('b (cg c s) l -> b (l cg) (c s)', cg = channels_group, s = patch_dim),
            nn.Linear(patch_dim * conv_channels // channels_group, dim),
        )

        # self.to_target_embedding = nn.Sequential(
        #     nn.Unfold(kernel_size = patch_size, stride = stride_size, padding=padding_size), # (N, C, *) -> (N, C × ∏(kernel_size), L)
        #     Rearrange('b (c s) l -> b (l c) s', c = pred_len),
        #     nn.Linear(patch_dim, dim),
        # )

        self.src_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        # self.tgt_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.time_embedding = TemporalEmbedding(dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, depth)
        self.decoder = MLPDecoder(pred_len, dim, patch_size, n_hidden=mlp_dim, c_groups=channels_group)

        # self.transformer = nn.Transformer(d_model=dim, nhead=heads, num_encoder_layers=depth, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        # self.decoder = MLPDecoder(pred_len, dim, patch_size)

    def forward(self, x, x_mark):
        x = x.squeeze(1)
        res = repeat(torch.mean(x[:, -1:], dim=1), 'b h w -> b s h w', s=self.pred_len)
        
        convf = self.conv_features(x)
        src = self.to_patch_embedding(convf)
        # b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        src += self.src_pos_embedding

        time_embedding = self.time_embedding(x_mark)
        time_embedding = repeat(time_embedding, 'b h w -> b s h w', s=self.num_patches // self.channels_group)
        time_embedding = rearrange(time_embedding, 'b s (g h) w -> b s g h w', g=self.channels_group)
        time_embedding = torch.flatten(time_embedding, start_dim=1, end_dim=2).mean(dim=2).squeeze(2)
        src += time_embedding
        # x = self.dropout(x)

        x = self.transformerEncoder(src)
        x = self.decoder(x)

        return x + res
    
    def _process_one_batch(self, batch):
        x, y, x_mark = batch
        y_hat = self(x, x_mark)
        if self.pred_len == 1:
            y = y.unsqueeze(1)
        return y_hat, y
