import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from datasets.Milan import Milan

from utils.time_features import time_features
from utils.milan_data import get_indexes_of_train


class MilanSW(Milan):
    """ Milan Dataset in a sliding window fashion """
    def __init__(self, 
                 format: str = 'normal',
                 close_len: int = 3,
                 period_len: int = 3,
                 label_len: int = 12,
                 out_len: int = 1,
                 window_size: int = 11,
                 flatten: bool = True,
                 **kwargs):
        super(MilanSW, self).__init__(**kwargs)
        if format not in ['normal', 'informer', 'sttran', '3comp']:
            raise ValueError("format must be one of 'normal', 'informer', 'sttran', '3comp'")
        self.format = format
        self.close_len = close_len
        self.period_len = period_len
        self.label_len = label_len
        self.out_len = out_len
        self.flatten = flatten
        self.window_size = window_size

    def prepare_data(self):
        Milan.prepare_data(self)
    
    def setup(self, stage=None):
        Milan.setup(self, stage)

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train'), batch_size=self.batch_size, shuffle=False, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage):
        if self.format == 'default':
            dataset = MilanSlidingWindowDataset(data, input_len=self.close_len, window_size=self.window_size, flatten=self.flatten)
        elif self.format =='informer':
            dataset =  MilanSWInformerDataset(data, self.milan_timestamps[stage],  
                                          aggr_time=self.aggr_time, input_len=self.close_len, 
                                          window_size=self.window_size, label_len=self.label_len)
        elif self.format == 'sttran':
            dataset =  MilanSWStTranDataset(data, self.aggr_time, self.close_len, 
                                        self.period_len, self.out_len)
        elif self.format == '3comp':
            dataset =  MilanSW3CompDataset(data, self.aggr_time, self.close_len, 
                                       self.period_len, window_size=self.window_size, flatten=self.flatten)
        # print(f'{stage} dataset length: {len(dataset)}')
        return dataset
class MilanSlidingWindowDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 window_size: int = 11,
                 input_len: int = 12,
                 flatten: bool = True,):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.window_size = window_size
        self.input_len = input_len
        self.flatten = flatten
        pad_size = window_size // 2
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
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


class MilanSW3CompDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3, *,
                 window_size: int = 11,
                 flatten: bool = True,):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.time_level = aggr_time
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len
        self.flatten = flatten
        self.out_len = 1
        self.window_size = window_size
        pad_size = window_size // 2
        self.milan_data_pad = np.pad(self.milan_data,
                                    ((0, 0), (pad_size, pad_size),
                                    (pad_size, pad_size)),
                                    'constant', constant_values=0)

    def __len__(self):
        return (self.milan_data.shape[0] - self.in_len) * self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        out_start_idx = n_slice + self.in_len
        
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, self.close_len, self.period_len)
        spatial_window = (self.window_size, self.window_size)
        idx_grid_data = self.milan_data_pad[:, n_row:n_row+self.window_size,
                                        n_col:n_col+self.window_size]
        X = np.array([idx_grid_data[i] if i >= 0 else np.zeros(spatial_window) for i in indices], dtype=np.float32)
        if self.flatten:
            X = X.reshape((-1, spatial_window[0] * spatial_window[1]))
        Y = self.milan_data[out_start_idx: out_start_idx+self.out_len, n_row, n_col]
        return (X, Y)


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
        pad_size = window_size // 2
        self.window_size = window_size
        self.input_len = input_len
        self.label_len = label_len
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
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


class MilanSWStTranDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3,
                 out_len: int = 3,
                 K_grids = 20):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len
        self.out_len = out_len
        self.K_grids = K_grids

        self.curr_slice = -1
        self.grid_topk = None

    def __len__(self):
         return (self.milan_data.shape[0]-self.in_len-self.out_len+1) * self.milan_data.shape[1] * self.milan_data.shape[2]
    
    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        out_start_idx = n_slice + self.in_len

        if n_slice == self.curr_slice:
            grids_topk = self.grid_topk
        else:
            Xc = self.milan_data[out_start_idx-self.close_len: out_start_idx] # Xc
            Xc = Xc.reshape((Xc.shape[0], Xc.shape[1] * Xc.shape[2])).transpose(1, 0)
            Xc = torch.from_numpy(Xc)
            N, C = Xc.shape
            grid_map = self._grid_selection(Xc, self.K_grids)
            Xc_expand = Xc.unsqueeze(0).expand(N, N, C)
            grids_topk = Xc_expand.gather(1, grid_map.unsqueeze(2).expand((*grid_map.shape, C)))
            self.grid_topk = grids_topk
            self.curr_slice = n_slice
        Xs = grids_topk[self.milan_data.shape[1] * n_row + n_col]

        idx_grid_data = self.milan_data[:, n_row, n_col]
        Xc = idx_grid_data[out_start_idx-self.close_len: out_start_idx] # Xc
        indices = get_indexes_of_train('sttran', self.time_level, out_start_idx, self.close_len, self.period_len)
        Xp = [idx_grid_data[i] if i >= 0 else 0 for i in indices]
        Xp = np.stack(Xp, axis=0).astype(np.float32)
        Xp = Xp.reshape((self.period_len, self.close_len))
        Y = idx_grid_data[out_start_idx: out_start_idx+self.out_len]

        return Xc, Xp, Xs, Y # (c,), (p, c), (K, c), (c,)

    def _grid_selection(self, Xc, K):
        # According to the correlation matrix A(t) - Pearson’s correlation coefficient, 
        # for each grid, we sort its correlations with other grids in a 
        # descending order and select the ﬁrst K grids. 
        # The value of K can be chosen experimentally.
        # corr_matrix = torch.corrcoef(Xc)
        with np.errstate(divide='ignore', invalid='ignore'):
            ncov = np.corrcoef(Xc)
            ncov[np.isnan(ncov)] = -1
        return torch.topk(torch.from_numpy(ncov), k=K, dim=1).indices
