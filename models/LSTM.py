from torch import nn

from pytorch_lightning import LightningModule
from torch.nn import L1Loss

from models.STBase import STBase


class LSTMRegressor(STBase):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 n_features, 
                 emb_size,
                 hidden_size, 
                 seq_len, 
                 num_layers, 
                 dropout, 
                 is_input_embedding=True,
                 pred_len = 1, 
                 **kwargs
                 ):
        super(LSTMRegressor, self).__init__(**kwargs)
        self.n_features = n_features
        self.embedding_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_input_embedding = is_input_embedding
        self.save_hyperparameters()

        lstm_input_size = emb_size if self.is_input_embedding else n_features
        if is_input_embedding:
            self.input_embedding = nn.Linear(n_features, emb_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_input_embedding:
            x = self.input_embedding(x)
        x, _ = self.lstm(x) # lstm_out = (batch_size, seq_len, hidden_size)
        x = self.linear(x).squeeze(2)
        y_pred = x[:, -self.pred_len:]
        return y_pred
