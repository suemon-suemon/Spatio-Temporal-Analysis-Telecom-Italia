from matplotlib.pyplot import cla
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from models.attentions import (ScaledDotProductAttention,
                               SimplifiedScaledDotProductAttention)
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


class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y


class DAModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model, kernel_size, H, W)
        self.channel_attention_module=ChannelAttentionModule(d_model, kernel_size, H, W)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out


class PyramidDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.patch_emb_to_matrix = Rearrange('b (h w) d -> b d h w', h=10, w=10)
        # self.linear = nn.Linear(dim, 32 * 9)
        # # C H W: 2 x 10 x 10 -> 8 x 4 x 4
        self.conv1 = nn.Sequential(nn.Conv2d(2, 8, kernel_size=4, stride=2),
                                   DAModule(8, 3, 4, 4))
        
        self.conv2 = nn.Sequential(
                                   nn.Conv2d(4, 8, kernel_size=3),
                                   DAModule(8, 3, 8, 8))

        self.conv3 = nn.Sequential(
                                   nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2, padding=2),
                                   DAModule(16, 3, 16, 16))

        self.conv4 = nn.Sequential(
                                   nn.ConvTranspose2d(50, 32, kernel_size=3, stride=3),
                                   DAModule(32, 3, 30, 30))

        self.convF = nn.Sequential(
                                   nn.Conv2d(64, 32, kernel_size=1),
                                   DAModule(32, 5, 30, 30),
                                   nn.Conv2d(32, 16, kernel_size=1),
                                   DAModule(16, 3, 30, 30),
                                   nn.Conv2d(16, 1, kernel_size=1))

        self.upsample1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.upsampleConv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=1)

        # self.conv1_up = nn.Sequential(nn.Unflatten(1, (4, 4, 4)),
        #                            nn.Conv2d(4, 8, (2, 2), padding='same'),
        #                            DAModule(8, 3, 4, 4),
        #                            nn.ConvTranspose2d(8, 8, (2, 2), stride=2))
        # self.conv1 = nn.Sequential(nn.Conv2d(8, 8, (3, 3), padding='same'),
        #                            DAModule(8, 3, 8, 8))
        # self.linear2 = nn.Sequential(nn.Flatten(1, 2),
        #                              nn.Linear(150 * 2 * 4, 16 * 8 * 8),
        #                              nn.Unflatten(1, (16, 8, 8)))
        # self.conv2_up = nn.Sequential(nn.Conv2d(16, 32, (5, 5), padding='same'),
        #                            nn.ConvTranspose2d(32, 16, (2, 2), stride=2))
        # self.conv2 = nn.Sequential(nn.Conv2d(16, 16, (5, 5), padding='same'),
        #                            DAModule(16, 5, 16, 16))
        # self.linear3 = nn.Sequential(nn.Flatten(1, 2),
        #                              nn.Linear(150 * 4 * 4, 32*16*16),
        #                              nn.Unflatten(1, (32, 16, 16)))
        # self.conv3_up = nn.Sequential(nn.Conv2d(32, 32, (7, 7), padding='same'),
        #                            nn.ConvTranspose2d(32, 16, (2, 2), stride=2, padding=1))
        # self.conv3 = nn.Sequential(nn.Conv2d(16, 16, (7, 7), padding='same'),
        #                            DAModule(16, 7, 30, 30))
        # self.linear4 = nn.Sequential(nn.Flatten(1, 2),
        #                              nn.Linear(150*10 * 4, 32*30*30),
        #                              nn.Unflatten(1, (32, 30, 30)))
        # self.conv4 = nn.Sequential(nn.Conv2d(32, 32, (5, 5), padding='same'),
        #                            DAModule(32, 5, 30, 30))
        # self.lastconv = nn.Conv2d(32, 1, (1, 1), padding=0)


    def forward(self, x):
        # TODO relu position
        # x = self.linear(x)
        mt = self.patch_emb_to_matrix(x)
        
        l1 = self.conv1(mt[:, 0:2, :, :]) # B x 8 x 4 x 4
        l1up = F.relu(self.upsample1(l1)) # B x 8 x 8 x 8

        l2 = self.conv2(mt[:, 2:6, :, :]) # B x 8 x 8 x 8
        l2up = F.relu(self.upsample2(torch.cat((l2, l1up), dim=1))) # B x 16 x 16 x 16

        l3 = self.conv3(mt[:, 6:14, :, :]) # B x 16 x 16 x 16
        l3up = F.relu(self.upsampleConv(torch.cat((l3, l2up), dim=1))) # B x 32 x 30 x 30
        
        l4 = self.conv4(mt[:, 14:, :, :]) # B x 32 x 30 x 30
        l4F = self.convF(torch.cat((l4, l3up), dim=1)) # B x Out x 30 x 30
        return l4F

        # x1 = F.relu(self.conv1_up(x[:, 0, :]))
        # x1 = F.relu(torch.cat((x1, self.conv1(x1)), dim=1))

        # t2 = self.linear2(x[:, 1:, 0:8]) + x1
        # x2 = F.relu(self.conv2_up(t2))
        # x2 = F.relu(torch.cat((x2, self.conv2(x2)), dim=1))
        
        # t3 = self.linear3(x[:, 1:, 8:24]) + x2
        # x3 = F.relu(self.conv3_up(t3))
        # x3 = F.relu(torch.cat((x3, self.conv3(x3)), dim=1))
        
        # t4 = self.linear4(x[:, 1:, 24:]) + x3
        # x4 = F.relu(t4 + self.conv4(t4))
        # return self.lastconv(x4)


class MLPDecoder(nn.Module):
    def __init__(self, n_input, n_output=9, n_hidden=64):
        super(MLPDecoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(n_input, n_hidden),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(n_hidden, n_output))

    def forward(self, x):
        x = self.linear(x)
        img = rearrange(x, 'b (h w) (dh dw) -> b (h dh) (w dw)', h=10, dh=3)
        return img


class ViT_pyramid(STBase):
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
        super(ViT_pyramid, self).__init__(**kwargs)
        self.seq_len = close_len
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        stride_height, stride_width = pair(stride_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        channels = close_len + period_len
        num_patches = ((image_height - patch_height) // stride_height + 1) * ((image_width - patch_width) // stride_width + 1)
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = stride_size),
            Rearrange('b (c p) l -> b l (c p)', c = channels, p = patch_dim),
            nn.Linear(patch_dim * channels, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder = MLPDecoder(dim)

    def forward(self, img):
        img = img.squeeze(1)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x + torch.mean(img, dim=1)
