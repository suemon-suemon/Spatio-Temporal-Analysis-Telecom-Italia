import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datasets.Milan import Milan

from utils.time_features import time_features


class MilanSW(Milan):
    """ Milan Dataset in a sliding window fashion """
    def __init__(self, 
                 format: str = 'normal',
                 in_len: int = 12, 
                 label_len: int = 12,
                 flatten: bool = True,
                 **kwargs):
        super(MilanSW, self).__init__(**kwargs)
        if format not in ['normal', 'informer']:
            raise ValueError("format must be one of 'normal', 'informer'")
        self.format = format
        self.in_len = in_len
        self.label_len = label_len
        self.flatten = flatten

    def prepare_data(self):
        Milan.prepare_data(self)
    
    def setup(self, stage=None):
        Milan.setup(self, stage)

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train'), batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage):
        if self.format == 'normal':
            return MilanSlidingWindowDataset(data, input_len=self.in_len, flatten=self.flatten)
        elif self.format =='informer':
            return MilanSWInformerDataset(data, self.milan_timestamps[stage],  
                                          aggr_time=self.aggr_time, input_len=self.in_len, 
                                          label_len=self.label_len)

class MilanSlidingWindowDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 window_size: int = 11,
                 input_len: int = 12,
                 flatten: bool = True,):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.pad_size = window_size // 2
        self.window_size = window_size
        self.input_len = input_len
        self.flatten = flatten
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (self.pad_size, self.pad_size),
                                      (self.pad_size, self.pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        return (self.milan_data.shape[0] - self.input_len) * self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        X = self.milan_data_pad[n_slice:n_slice+self.input_len,
                                n_row:n_row+self.window_size,
                                n_col:n_col+self.window_size]
        if self.flatten:
            X = X.reshape((self.input_len, self.window_size * self.window_size))
        return (X, self.milan_data[n_slice+self.input_len, n_row, n_col].reshape(-1))


class MilanSWInformerDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 timestamps: pd.DataFrame,
                 aggr_time = None,
                 window_size: int = 11,
                 input_len: int = 12,
                 label_len: int = 12):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.timestamps = time_features(timestamps, timeenc=1,
                                        freq='h' if aggr_time == 'hour' else 't')
        self.pad_size = window_size // 2
        self.window_size = window_size
        self.input_len = input_len
        self.label_len = label_len
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (self.pad_size, self.pad_size),
                                      (self.pad_size, self.pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        return (self.milan_data.shape[0] - self.input_len) * self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        X = self.milan_data_pad[n_slice:n_slice+self.input_len,
                                n_row:n_row+self.window_size,
                                n_col:n_col+self.window_size]
        X_timefeature = self.timestamps[n_slice:n_slice+self.input_len]
        Y_timefeature = self.timestamps[n_slice+self.input_len-self.label_len: n_slice+self.input_len+1]
        X = X.reshape((self.input_len, self.window_size * self.window_size))
        Y = self.milan_data[n_slice+self.input_len-self.label_len: n_slice+self.input_len+1, n_row, n_col].reshape(-1, 1)
        return X, Y, X_timefeature, Y_timefeature 