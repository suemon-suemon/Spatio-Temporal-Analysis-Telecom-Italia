from sklearn.feature_extraction import img_to_graph
from models.STBase import STBase
from torch.nn import Conv3d, Linear, Sequential, ReLU, Flatten
from torch import concat

from models.convlstm import ConvLSTM

class STN(STBase):
    '''
    Long-term mobile trafﬁc forecasting using deep spatio-temporal neural networks
    '''
    def __init__(self, 
                 x_dim: int = 11,
                 y_dim: int = 11, 
                 seq_len: int = 12, 
                 **kwargs
                 ):
        super(STN, self).__init__(**kwargs)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.seq_len = seq_len
        self.save_hyperparameters()

        img_size = (x_dim, y_dim)
        self.conv3d1 = Conv3d(in_channels=1, out_channels=3, kernel_size=(3, 3, 3), padding='same')
        self.convlstm1 = ConvLSTM(img_size, input_dim=1, hidden_dim=3, kernel_size=(3, 3), batch_first=True)

        self.conv3d2 = Conv3d(in_channels=6, out_channels=6, kernel_size=(3, 3, 3), padding='same')
        self.convlstm2 = ConvLSTM(img_size, input_dim=6, hidden_dim=6, kernel_size=(3, 3), batch_first=True)
        
        self.conv3d3 = Conv3d(in_channels=12, out_channels=12, kernel_size=(3, 3, 3), padding='same')
        self.convlstm3 = ConvLSTM(img_size, input_dim=12, hidden_dim=12, kernel_size=(3, 3), batch_first=True)
        
        self.decoder = Sequential(
            Flatten(),
            Linear(24*12*11*11, 4096),
            ReLU(),
            Linear(4096, 1024),
            ReLU(),
            Linear(1024, 1)
        )

    def forward(self, x):
        # SHAPE: (batch_size, 1, seq_len, x_dim, y_dim)
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))
        conv3d1 = self.conv3d1(x)
        convlstm1, _, _ = self.convlstm1(x)
        x = concat([conv3d1, convlstm1], dim=1) 
        conv3d2 = self.conv3d2(x)
        convlstm2, _, _ = self.convlstm2(x)
        x = concat([conv3d2, convlstm2], dim=1) 
        conv3d3 = self.conv3d3(x)
        convlstm3, _, _ = self.convlstm3(x)
        x = concat([conv3d3, convlstm3], dim=1)
        x = self.decoder(x)
        return x
