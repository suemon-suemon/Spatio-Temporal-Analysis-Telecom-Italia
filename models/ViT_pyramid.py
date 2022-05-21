import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.STBase import STBase

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
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


class PyramidDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Unflatten(2, (4, 4)),
                                   nn.Conv2d(1, 4, (1, 1), padding=0),
                                   nn.ConvTranspose2d(4, 4, (2, 2), stride=2),
                                   nn.Conv2d(4, 8, (1, 1), padding=0),
                                   nn.ReLU())
        self.linear2 = nn.Sequential(nn.Flatten(1, 2),
                                     nn.Linear(10 * 16, 8 * 8 * 8),
                                     nn.Unflatten(1, (8, 8, 8)))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, (1, 1), padding=0),
                                   nn.ConvTranspose2d(16, 16, (2, 2), stride=2),
                                   nn.Conv2d(16, 8, (1, 1), padding=0),
                                   nn.ReLU())
        self.linear3 = nn.Sequential(nn.Flatten(1, 2),
                                     nn.Linear(40*16, 8*16*16),
                                     nn.Unflatten(1, (8, 16, 16)))
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, (1, 1), padding=0),
                                   nn.ConvTranspose2d(8, 8, (2, 2), stride=2, padding=1),
                                   nn.Conv2d(8, 2, (1, 1), padding=0),
                                   nn.ReLU())
        self.linear4 = nn.Sequential(nn.Flatten(1, 2),
                                     nn.Linear(100*16, 2*30*30),
                                     nn.Unflatten(1, (2, 30, 30)))
        self.conv4 = nn.Conv2d(2, 1, (1, 1), padding=0)
        
    def forward(self, x):
        x1 = self.conv1(x[:, 0:1, :]) # output shape 4, 8, 8
        t2 = self.linear2(x[:, 1:11, :]) # output shape 4, 8, 8
        x2 = self.conv2(x1 + t2)
        t3 = self.linear3(x[:, 11:51, :])
        x3 = self.conv3(x2 + t3)
        t4 = self.linear4(x[:, 51:, :])
        x4 = self.conv4(x3 + t4)
        return x4

class ViT(STBase):
    def __init__(self, *, image_size, # (11, 11)
                          patch_size, # (3, 3)
                          stride_size, # (2, 2)
                          dim, 
                          depth, 
                          heads, 
                          mlp_dim, 
                          pred_len = 1, 
                          pool = 'cls', 
                          close_len = 3, 
                          period_len = 0,
                          dim_head = 64, 
                          dropout = 0., 
                          emb_dropout = 0.,
                          **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.seq_len = close_len
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        stride_height, stride_width = pair(stride_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        channels = close_len + period_len
        num_patches = channels * ((image_height - patch_height) // stride_height + 1) * ((image_width - patch_width) // stride_width + 1)
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = stride_size),
            Rearrange('b (c p) l -> b (l c) p', c = channels, p = patch_dim),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder = PyramidDecoder(dim, None)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim//2),
        #     nn.ReLU(),
        #     nn.Linear(dim//2, pred_len),
        # )

    def forward(self, img):
        img = img.squeeze(1)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        # return self.mlp_head(x)