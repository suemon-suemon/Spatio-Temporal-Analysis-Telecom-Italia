import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dtaidistance import dtw

from datasets.Milan import Milan
from utils.time_features import time_features
from utils.milan_data import get_indexes_of_train
from torch.utils.data.sampler import RandomSampler

class MilanFG(Milan):
    """ Milan Dataset in a full-grid fashion """
    def __init__(self, 
                 format: str = 'default',
                 close_len: int = 12, 
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 0,
                 pred_len: int = 1,
                 **kwargs):
        super(MilanFG, self).__init__(**kwargs)
        self.format = format
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.label_len = label_len
        self.pred_len = pred_len

    def prepare_data(self):
        Milan.prepare_data(self)
    
    def setup(self, stage=None):
        Milan.setup(self, stage)
        train_len, val_len, test_len = self.get_default_len(self.time_range)
        self.milan_timestamps = {
            "train": self.timestamps[:train_len],
            "val": self.timestamps[train_len:train_len+val_len],
            "test": self.timestamps[train_len+val_len-(self.close_len+self.pred_len-1):train_len+val_len+test_len],
        }
        self.milan_train, self.milan_val, self.milan_test = self.train_test_split(self.milan_grid_data, train_len, val_len, test_len)
        self.milan_test = np.concatenate((self.milan_val[-(self.close_len+self.pred_len-1):], self.milan_test))
        print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train', self.meta), batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val', self.meta), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test', self.meta), batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage, meta=None):
        if self.format == 'default':
            return MilanFullGridDataset(data, self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.pred_len)
        elif self.format == 'informer':
            return MilanFGInformerDataset(data, self.milan_timestamps[stage], self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.label_len, self.pred_len)
        elif self.format == 'sttran':
            return MilanFGStTranDataset(data, self.aggr_time, self.close_len, self.period_len, self.pred_len)
        elif self.format == 'stgcn':
            return MilanFGStgcnDataset(data, self.aggr_time, self.close_len, self.period_len, self.trend_len, self.pred_len)
        elif self.format == 'timeF':
            return MilanFGTimeFDataset(data, meta, self.milan_timestamps[stage], self.aggr_time, self.close_len, self.period_len, self.trend_len, self.pred_len)


class MilanFullGridDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len
        print("MilanFullGridDataset: length {}".format(self.__len__()))

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)

        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0).astype(np.float32)
        # X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len].astype(np.float32)
        return X, Y


class MilanFGInformerDataset(Dataset):
    def __init__(self,
                 milan_data,
                 timestamps,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 12,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.timestamps = time_features(timestamps, timeenc=1, 
                                        freq='h' if self.time_level == 'hour' else 't')
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0)
        X = X.reshape((X.shape[0], X.shape[2] * X.shape[3])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx-self.label_len: out_start_idx+self.pred_len].squeeze()
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

        X_timefeature = [self.timestamps[i] if i >= 0 else np.zeros((self.timestamps.shape[1])) for i in indices]
        X_timefeature = np.stack(X_timefeature, axis=0)
        Y_timefeature = self.timestamps[out_start_idx-self.label_len: out_start_idx+self.pred_len]

        return X, Y, X_timefeature, Y_timefeature
    

class MilanFGTimeFDataset(Dataset):
    def __init__(self,
                 milan_data,
                 meta,
                 timestamps,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.meta = meta

        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len
        self.timestamps = time_features(timestamps, timeenc=1, 
                                freq='h' if self.time_level == 'hour' else 't')
        print("MilanFullGridDataset: length {}".format(self.__len__()))

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len, pred_len=self.pred_len)

        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0).astype(np.float32)
        # dist = dtw.distance_matrix_fast(X.reshape(self.in_len, -1).T.astype(np.float64))
        # X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len].astype(np.float32)

        X_timefeature = [self.timestamps[i] if i >= 0 else np.zeros((self.timestamps.shape[1])) for i in indices]
        X_timefeature = np.stack(X_timefeature, axis=0).astype(np.float32)
        X_meta = self.meta.astype(np.float32)
        return X, Y, X_timefeature, X_meta


class MilanFGStTranDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3,
                 pred_len: int = 3):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]

        Xc = self.milan_data[out_start_idx-self.close_len: out_start_idx] # Xc
        indices = get_indexes_of_train('sttran', self.time_level, out_start_idx, self.close_len, self.period_len)
        Xp = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        Xp = np.stack(Xp, axis=0).astype(np.float32)
        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len]

        Xc = Xc.reshape((Xc.shape[0], Xc.shape[1] * Xc.shape[2])).transpose(1, 0)
        Xp = Xp.reshape((self.period_len, self.close_len, Xp.shape[1] * Xp.shape[2])).transpose(2, 1, 0)
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2])).transpose(1, 0)
        return Xc, Xp, Y # (N, c), (N, p, c), (N, c)


class MilanFGStgcnDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)
        indices.reverse()
        
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0).astype(np.float32)
        X = X.reshape((X.shape[0], 1, X.shape[1] * X.shape[2])) # (n_timestamps, n_features, n_grid_row, n_grid_col)))
        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len]

        Xc = X[:self.close_len].transpose(2, 1, 0)
        if self.period_len == 0 and self.trend_len == 0:
            return [Xc], Y
        elif self.trend_len == 0:
            Xp = X[self.close_len: self.close_len+self.period_len].transpose(2, 1, 0)
            return [Xc, Xp], Y
        else:
            Xp = X[self.close_len: self.close_len+self.period_len].transpose(2, 1, 0)
            Xt = X[self.close_len+self.period_len: self.close_len+self.period_len+self.trend_len].transpose(2, 1, 0)
            return [Xc, Xp, Xt], Y