from datasets.Milan import Milan
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from pytorch_lightning import LightningDataModule


class MilanFG(Milan, LightningDataModule):
    """ Milan Dataset in a full-grid fashion """
    def __init__(self, 
                 close_len: int = 12, 
                 period_len: int = 0,
                 trend_len: int = 0,
                 **kwargs):
        Milan.__init__(self, **kwargs)
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len

    def prepare_data(self):
        super().prepare_data()
    
    def setup(self, stage=None):
        super().setup(stage)

    def train_dataloader(self):
        # TODO: fix parameters!
        milan_train_ds = MilanFullGridDataset(self.milan_train, self.aggr_time, self.close_len, 
                                                   self.period_len, self.trend_len, self.out_len)
        print("Length of milan training dataset: ", len(milan_train_ds))
        return DataLoader(milan_train_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        milan_val_ds = MilanFullGridDataset(self.milan_val, self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.out_len)
        return DataLoader(milan_val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        milan_test_ds = MilanFullGridDataset(self.milan_test, self.aggr_time, self.close_len, 
                                             self.period_len, self.trend_len, self.out_len)
        return DataLoader(milan_test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        milan_pred_ds = MilanFullGridDataset(self.milan_test, self.aggr_time, self.close_len, 
                                             self.period_len, self.trend_len, self.out_len)
        return DataLoader(milan_pred_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)


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