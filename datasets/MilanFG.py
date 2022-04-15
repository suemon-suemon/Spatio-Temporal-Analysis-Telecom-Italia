import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.Milan import Milan
from utils.time_features import time_features


class MilanFG(Milan):
    """ Milan Dataset in a full-grid fashion """
    def __init__(self, 
                 format: str = 'normal',
                 close_len: int = 12, 
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 12,
                 **kwargs):
        super(MilanFG, self).__init__(**kwargs)
        self.format = format
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.label_len = label_len

    def prepare_data(self):
        super().prepare_data(self)
    
    def setup(self, stage=None):
        super().setup(self, stage)

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val'), batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test'), batch_size=self.batch_size, shuffle=False, num_workers=8)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage):
        if self.format == 'normal':
            return MilanFullGridDataset(data, self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.out_len)
        elif self.format =='informer':
            return MilanFGInformerDataset(data, self.milan_timestamps[stage], self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.label_len, self.out_len)


class MilanFullGridDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 out_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.is_hour_level = (aggr_time == 'hour')
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.out_len = out_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.out_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        Y = self.milan_data[out_start_idx: out_start_idx+self.out_len].squeeze()
        X = self.milan_data[out_start_idx-self.close_len: out_start_idx] # Xc
        Xp = [self._get_data_days_ahead(self.milan_data, out_start_idx, i, self.is_hour_level) 
                for i in reversed(range(1, self.period_len+1))]
        Xt = [self._get_data_weeks_ahead(self.milan_data, out_start_idx, i, self.is_hour_level) 
                for i in reversed(range(1, self.trend_len+1))]
        Xp = np.stack(Xp, axis=0) if len(Xp) > 0 else None
        Xt = np.stack(Xt, axis=0) if len(Xt) > 0 else None
        if Xp is not None:
            X = np.concatenate([X, Xp], axis=0)
        if Xt is not None:
            X = np.concatenate([X, Xt], axis=0)
        X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y

    # @staticmethod
    # def _get_data_hrs_ahead(data, idx, n_hrs):
    #     # is hour level must be False, if using this function
    #     TIME_STEPS_OF_HOUR = 6 # 1 hours = 6 * 10 mins
    #     ahead_idx = idx - n_hrs * TIME_STEPS_OF_HOUR
    #     if ahead_idx < 0:
    #         return np.zeros(data[0].shape)
    #     else:
    #         return data[ahead_idx]
    
    @staticmethod
    def _get_data_days_ahead(data, idx, n_days, is_hour_level=False):
        TIME_STEPS_OF_DAY = 24 if is_hour_level else 144 # 1 day = 24 * 6 * 10 mins
        ahead_idx = idx - n_days * TIME_STEPS_OF_DAY
        if ahead_idx < 0:
            return np.zeros(data[0].shape)
        else:
            return data[ahead_idx]
    
    @staticmethod
    def _get_data_weeks_ahead(data, idx, n_weeks, is_hour_level=False):
        TIME_STEPS_OF_WEEK = 168 if is_hour_level else 1008 # 168 = 24 * 7; 1008 = 24 * 7 * 6
        ahead_idx = idx - n_weeks * TIME_STEPS_OF_WEEK
        if ahead_idx < 0:
            return np.zeros(data[0].shape)
        else:
            return data[ahead_idx]


class MilanFGInformerDataset(Dataset):
    def __init__(self,
                 milan_data,
                 timestamps,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 12,
                 out_len: int = 1):
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
        self.out_len = out_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.out_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = _get_indexes_of_train('informer', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx-self.label_len: out_start_idx+self.out_len].squeeze()
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

        X_timefeature = [self.timestamps[i] if i >= 0 else np.zeros((self.timestamps.shape[1])) for i in indices]
        X_timefeature = np.stack(X_timefeature, axis=0)
        Y_timefeature = self.timestamps[out_start_idx-self.label_len: out_start_idx+self.out_len]

        return X, Y, X_timefeature, Y_timefeature
    

class MilanFGStTranDataset(Dataset):
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3,
                 out_len: int = 3):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len
        self.out_len = out_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.out_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]

        Y = self.milan_data[out_start_idx: out_start_idx+self.out_len].squeeze()
        Xc = self.milan_data[out_start_idx-self.close_len: out_start_idx] # Xc
        indices = _get_indexes_of_train('sttran', self.time_level, out_start_idx, self.close_len, self.period_len)
        Xp = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        Xp = np.stack(Xp, axis=0)

        Xc = Xc.reshape((Xc.shape[0], Xc.shape[1] * Xc.shape[2])).transpose(1, 0)
        Xp = Xp.reshape((self.period_len, self.close_len, Xp.shape[1] * Xp.shape[2])).transpose(2, 1, 0)
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2])).transpose(1, 0)
        return Xc, Xp, Y # (N, c), (N, p, c), (N, c)


def _get_indexes_of_train(format, time_level, out_start_idx, close_len, period_len, trend_len = 0):
    if time_level == 'hour':
        TIME_STEPS_OF_DAY = 24
    else: # 10 mins level
        TIME_STEPS_OF_DAY = 24 * 6
    indices = []
    if format == 'informer':
        indices += [out_start_idx-i-1 for i in range(close_len)]
        if period_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY-1 for i in range(period_len)]
        if trend_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY*7 for i in range(trend_len)]
    elif format == 'sttran':
        if period_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY-j for j in range(close_len) for i in range(period_len)]
    indices.reverse()
    return indices