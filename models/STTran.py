
import torch
import torch.nn as nn
import math

from models.STBase import STBase

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class STTran(STBase):
    def __init__(self, 
                 close_len = 3,
                 period_len = 3, 
                 out_len = 3,
                 k_grids = 9,
                 q_grids = 9,
                 n_grids = 900,
                 sp_dmodel = 64,
                 cl_dmodel = 256,
                 pe_dmodel = 128,
                 **kwargs):
        super(STTran, self).__init__(**kwargs)   
        self.seq_len = close_len
        self.out_len = out_len
        self.k_grids = k_grids
        self.save_hyperparameters()

        # TODO: try to sparse linear
        # Spatial Encoding
        self.sp_pos_encoder = PositionalEncoding(sp_dmodel)
        self.sp_target_embedding = nn.Linear(n_grids * close_len, sp_dmodel)
        self.sp_linear = nn.Linear(sp_dmodel, n_grids * out_len)
        self.spatial_tran = nn.Transformer(d_model=sp_dmodel, nhead=8, batch_first=True) # default layers 6
        # Close Encoding
        self.cl_pos_encoder = PositionalEncoding(cl_dmodel)
        self.cl_target_embedding = nn.Linear(n_grids * q_grids, cl_dmodel)
        self.cl_init_embedding = nn.Linear(n_grids * close_len, cl_dmodel)
        self.cl_linear = nn.Linear(cl_dmodel, n_grids * out_len)
        self.close_tran = nn.Transformer(d_model=cl_dmodel, nhead=8, batch_first=True)
        # Period Encoding
        self.pe_pos_encoder = PositionalEncoding(pe_dmodel)
        self.pe_target_embedding = nn.Linear(n_grids * close_len, pe_dmodel)
        self.pe_linear = nn.Linear(pe_dmodel, n_grids * out_len)
        self.period_tran = nn.Transformer(d_model=pe_dmodel, nhead=8, batch_first=True)
        # Temporal Fusion
        # self.fusion_linear = nn.Linear(cl_dmodel + pe_dmodel, 1)
        self.sp_gate = nn.Linear(sp_dmodel, 1)
        self.cl_gate = nn.Linear(cl_dmodel, 1)
        self.pe_gate = nn.Linear(pe_dmodel, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, xc, xp):
        (B, N, P, C) = xp.shape
        K, O = self.k_grids, self.out_len
        grid_map = self._grid_selection(xc, self.k_grids)
        xc_expand = xc.unsqueeze(2).expand(B, N, N, C)
        topk_grids = xc_expand.gather(2, grid_map.unsqueeze(3).expand((*grid_map.shape, C)))
        
        # Spatial Encoding
        xs_src = topk_grids.transpose(2, 1).reshape(B, K, -1) # (B, K, N*C)
        xs_tgt = (xc.unsqueeze(1)).reshape(B, 1, -1) # (B, 1, N*C)
        xs_src = self.sp_target_embedding(xs_src)
        xs_tgt = self.sp_target_embedding(xs_tgt)
        sp_out = self.spatial_tran(xs_src, xs_tgt).squeeze()
        sp_pred = self.sp_linear(sp_out).view(B, N, O)

        # Close Encoding
        xc_src = topk_grids.permute(0, 3, 1, 2).reshape(B, C, -1) # (B, C, N*K)
        xc_tgt = xp.mean(dim=2).reshape(B, 1 ,-1) # (B, 1, N*C)
        xc_src = self.cl_target_embedding(xc_src)
        xc_tgt = self.cl_init_embedding(xc_tgt)
        xc_src = self.cl_pos_encoder(xc_src)
        cl_out = self.close_tran(xc_src, xc_tgt).squeeze()
        cl_pred = self.cl_linear(cl_out).view(B, N, O)

        # Period Encoding
        xp_src = xp.transpose(2, 1).reshape(B, P, -1) # (B, P, N*C)
        xp_tgt = (xc.unsqueeze(1)).reshape(B, 1, -1) # (B, 1, N*C)
        xp_src = self.pe_target_embedding(xp_src)
        xp_tgt = self.pe_target_embedding(xp_tgt)
        xp_tgt = self.pe_pos_encoder(xp_tgt)
        pe_out = self.period_tran(xp_src, xp_tgt).squeeze()
        pe_pred = self.pe_linear(pe_out).view(B, N, O)

        # Temporal Fusion
        # gate = torch.sigmoid(self.fusion_linear(torch.cat([cl_out, pe_out], dim=1))).unsqueeze(2)
        # pred = gate * cl_pred + (1 - gate) * pe_pred + sp_pred

        cl_gate = self.cl_gate(cl_out)
        pe_gate = self.pe_gate(pe_out)
        sp_gate = self.sp_gate(sp_out)
        gates = torch.stack([cl_gate, pe_gate, sp_gate], 1).squeeze()
        atts = self.softmax(gates).unsqueeze(2)
        pred = atts[:, [0]] * cl_pred + atts[:, [1]] * pe_pred + atts[:, [2]] * sp_pred
        return pred

    def _process_one_batch(self, batch):
        xc, xp, y = batch
        y_hat = self(xc, xp)
        return y_hat, y

    def _grid_selection(self, Xc, K):
        # According to the correlation matrix A(t) - Pearson’s correlation coefficient, 
        # for each grid, we sort its correlations with other grids in a 
        # descending order and select the ﬁrst K grids. 
        # The value of K can be chosen experimentally.
        # corr_matrix = torch.corrcoef(Xc)
        (_, _, N) = Xc.shape # batch_size, n_grids, seq_len
        mean = Xc.mean(dim=2).unsqueeze(2)
        diffs = Xc - mean.expand_as(Xc)
        prods = torch.bmm(diffs, diffs.transpose(1, 2))
        bcov = prods / (N - 1)
        norm = torch.bmm(Xc.std(dim=2).unsqueeze(2), Xc.std(dim=2).unsqueeze(1))
        ncov = bcov.div(norm)
        return torch.topk(ncov, k=K, dim=2).indices
