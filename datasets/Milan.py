import os
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from utils.load_data import load_and_save_telecom_data_by_tele, load_part_grid_data


class Milan(LightningDataModule):
    def __init__(self, 
                 data_dir: str = 'data/sms-call-internet-mi',
                 in_len: int = 12, 
                 out_len: int = 1, 
                 batch_size: int = 64, 
                 tele_column: str = 'internet',):
        super().__init__()
        self.data_dir = data_dir
        assert tele_column in ['internet',
                               'smsin', 'smsout', 'callin', 'callout']
        self.tele_column = tele_column
        self.batch_size = batch_size
        self.in_len = in_len
        self.out_len = out_len

        self.dataset_start_date = {'year': 2013, 'month': 11, 'day': 1}
        self.dataset_end_date = {'year': 2013, 'month': 11, 'day': 31}
        self.val_split_date = Milan.get_default_split_date()['val']
        self.test_split_date = Milan.get_default_split_date()['test']

    @staticmethod
    def train_test_split(data: np.ndarray, train_len: int, val_len: int=None, is_val=True) -> tuple:
        train = data[:train_len, :]
        if is_val:
            val = data[train_len:train_len+val_len, :]
            test = data[train_len+val_len:, :]
            return train, val, test
        else:
            test = data[train_len:, :]
            return train, test

    @staticmethod
    def get_default_split_date() -> dict:
        # return val and test split date
        return {'val': {'year': 2013, 'month': 11, 'day': 17}, 'test': {'year': 2013, 'month': 11, 'day': 21}}

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, 'milan_telecom_data.csv.gz')):
            start_date = self.dataset_start_date['day']
            end_date = self.dataset_end_date['day']
            if end_date > 10:
                paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) for i in range(start_date, 10)] +\
                        ['sms-call-internet-mi-2013-11-{i}.csv'.format(
                            i=i) for i in range(10, end_date)]
            else:
                paths = [
                    'sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) for i in range(start_date, end_date)]
            paths = [os.path.join(self.data_dir, path) for path in paths]
            load_and_save_telecom_data_by_tele(
                paths, self.data_dir, tele_column=self.tele_column)
        else:
            print('milan_telecom_data.csv.gz already exists in {}'.format(
                self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        milan_grid_data, milan_df = load_part_grid_data(self.data_dir, col1=41, col2=70, row1=41, row2=70)
        train_len = milan_df['time'][milan_df['time'] < pd.Timestamp(
            **self.val_split_date)].unique().shape[0]
        val_len = milan_df['time'][(milan_df['time'] >= pd.Timestamp(
            **self.val_split_date)) & (milan_df['time'] < pd.Timestamp(**self.test_split_date))].unique().shape[0]
        milan_train, milan_val, milan_test = Milan.train_test_split(milan_grid_data, train_len, val_len)
        self.milan_train = milan_train.reshape(-1, 30, 30)
        self.milan_val = milan_val.reshape(-1, 30, 30)
        self.milan_test = milan_test.reshape(-1, 30, 30)

    def train_dataloader(self):
        milan_train = MilanDataset(self.milan_train, segment_size=self.in_len)
        print("length of milan training dataset: ", len(milan_train))
        return DataLoader(milan_train, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(MilanDataset(self.milan_val, segment_size=self.in_len), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(MilanDataset(self.milan_test, segment_size=self.in_len), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(MilanDataset(self.milan_test, segment_size=self.in_len), batch_size=self.batch_size, shuffle=False, num_workers=4)


class MilanDataset(Dataset):
    def __init__(self,
                 milan_data: pd.DataFrame,
                 window_size: int = 11,
                 segment_size: int = 12):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        self.milan_data = milan_data
        self.pad_size = window_size // 2
        self.window_size = window_size
        self.segment_size = segment_size
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (self.pad_size, self.pad_size),
                                      (self.pad_size, self.pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        return (self.milan_data.shape[0] - self.segment_size) * self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        n_slice = index // (self.milan_data.shape[1]
                            * self.milan_data.shape[2])
        n_row = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (
            self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        X = self.milan_data_pad[n_slice:n_slice+self.segment_size,
                                n_row:n_row+self.window_size,
                                n_col:n_col+self.window_size]
        X = X.reshape((self.segment_size, self.window_size * self.window_size))
        return (X, self.milan_data[n_slice+self.segment_size, n_row, n_col].reshape(-1))
