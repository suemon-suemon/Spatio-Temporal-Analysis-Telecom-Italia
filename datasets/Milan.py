import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from utils.load_data import (load_and_save_telecom_data_by_tele,
                             load_part_grid_data)


class Milan():
    def __init__(self,
                 data_dir: str = 'data/sms-call-internet-mi',
                 aggr_time: str = None,
                 out_len: int = 1,
                 batch_size: int = 64,
                 normalize: bool = False,
                 tele_column: str = 'internet',
                 file_name: str = 'milan_telecom_data.csv.gz',
                 ):
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        if tele_column not in ['internet', 'smsin', 'smsout', 'callin', 'callout']:
            raise ValueError('tele_column must be one of internet, smsin, smsout, callin, callout')
        self.aggr_time = aggr_time
        self.tele_column = tele_column
        
        self.data_dir = data_dir
        self.file_name = file_name
        self.normalize = normalize

        self.batch_size = batch_size
        self.out_len = out_len

        self.dataset_start_date = {'year': 2013, 'month': 11, 'day': 1}
        self.dataset_end_date = {'year': 2013, 'month': 11, 'day': 31}
        self.val_split_date = self.get_default_split_date()['val']
        self.test_split_date = self.get_default_split_date()['test']

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
        return {'val': {'year': 2013, 'month': 11, 'day': 17}, 
                'test': {'year': 2013, 'month': 11, 'day': 21}}

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            start_date = self.dataset_start_date['day']
            end_date = self.dataset_end_date['day']
            if end_date > 10:
                paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) 
                         for i in range(start_date, 10)] + \
                        ['sms-call-internet-mi-2013-11-{i}.csv'.format(i=i) 
                         for i in range(10, end_date)]
            else:
                paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) 
                         for i in range(start_date, end_date)]
            paths = [os.path.join(self.data_dir, path) for path in paths]
            load_and_save_telecom_data_by_tele(paths, self.data_dir, tele_column=self.tele_column)
        else:
            print('{} already exists in {}'.format(self.file_name, self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            raise FileNotFoundError('{} not found in {}'.format(self.file_name, self.data_dir))
        milan_grid_data, milan_df = load_part_grid_data(self.data_dir, self.file_name, aggr_time=self.aggr_time, normalize=self.normalize, col1=41, col2=70, row1=41, row2=70)
        train_len = milan_df['time'][milan_df['time'] < pd.Timestamp(
            **self.val_split_date)].unique().shape[0]
        val_len = milan_df['time'][(milan_df['time'] >= pd.Timestamp(
            **self.val_split_date)) & (milan_df['time'] < pd.Timestamp(**self.test_split_date))].unique().shape[0]
        milan_train, milan_val, milan_test = Milan.train_test_split(milan_grid_data, train_len, val_len)
        self.milan_train = milan_train.reshape(-1, 30, 30)
        self.milan_val = milan_val.reshape(-1, 30, 30)
        self.milan_test = milan_test.reshape(-1, 30, 30)