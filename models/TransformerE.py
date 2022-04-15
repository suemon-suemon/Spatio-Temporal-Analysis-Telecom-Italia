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
          

class TransformerEncoder(STBase):
    def __init__(self, 
                 seq_len,
                 d_input, 
                 d_model=128,
                 num_layers=2,
                 dropout=0.2,
                 **kwargs):
        kwargs['reduceLRPatience'] = 2
        super(TransformerEncoder, self).__init__(**kwargs)   
        self.seq_len = seq_len     
        self.src_mask = None
        self.dense_shape = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(d_model * seq_len, 1)
        self.save_hyperparameters()

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).type_as(src)
            self.src_mask = mask
        src = self.dense_shape(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.reshape(output.size(0), -1)
        output = self.decoder(output)
        return torch.sigmoid(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask