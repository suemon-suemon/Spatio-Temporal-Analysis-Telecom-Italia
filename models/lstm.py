import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LSTMRegressor(LightningModule):
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
                learning_rate,
                criterion,
                is_input_embedding=True):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.embedding_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.is_input_embedding = is_input_embedding

        lstm_input_size = emb_size if self.is_input_embedding else n_features
        if is_input_embedding:
            self.input_embedding = nn.Linear(n_features, emb_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.save_hyperparameters()
        
    def forward(self, x):
        if self.is_input_embedding:
            x = self.input_embedding(x)
        x, _ = self.lstm(x) # lstm_out = (batch_size, seq_len, hidden_size)
        y_pred = self.linear(x[:,-1])
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat