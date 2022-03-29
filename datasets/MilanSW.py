import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datasets.Milan import Milan
from pytorch_lightning import LightningDataModule


class MilanSW(Milan, LightningDataModule):
    """ Milan Dataset in a sliding window fashion """
    def __init__(self, 
                 data_dir: str = 'data/sms-call-internet-mi',
                 aggr_time: str = None,
                 in_len: int = 12, 
                 out_len: int = 1, 
                 batch_size: int = 64, 
                 tele_column: str = 'internet',
                 normalize: bool = False,):
        Milan.__init__(self, data_dir, aggr_time, out_len, batch_size, normalize, tele_column)
        self.in_len = in_len

    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage=None):
        return super().setup(stage)

    def train_dataloader(self):
        milan_train = MilanSlidingWindowDataset(self.milan_train, input_len=self.in_len)
        print("Length of milan training dataset: ", len(milan_train))
        return DataLoader(milan_train, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(MilanSlidingWindowDataset(self.milan_val, input_len=self.in_len), 
                          batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(MilanSlidingWindowDataset(self.milan_test, input_len=self.in_len), 
                          batch_size=self.batch_size, shuffle=False, num_workers=4)


class MilanSlidingWindowDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 window_size: int = 11,
                 input_len: int = 12):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.pad_size = window_size // 2
        self.window_size = window_size
        self.input_len = input_len
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (self.pad_size, self.pad_size),
                                      (self.pad_size, self.pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        return (self.milan_data.shape[0] - self.input_len) * self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1]
                            * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        X = self.milan_data_pad[n_slice:n_slice+self.input_len,
                                n_row:n_row+self.window_size,
                                n_col:n_col+self.window_size]
        X = X.reshape((self.input_len, self.window_size * self.window_size))
        return (X, self.milan_data[n_slice+self.input_len, n_row, n_col].reshape(-1))
