import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.load_data import load_and_save_telecom_data_by_tele
from utils.milano_grid import map_back


def _gen_cellids_by_colrow(grid_range) -> list:
    """
    Generate cellid list by col and row
    :param grid_range: the range of grids, (row_min, row_max, col_min, col_max)
    """
    row1, row2, col1, col2 = grid_range
    cellids = []
    for col in range(col1, col2 + 1):
        for row in range(row1, row2 + 1):
            cellids.append(map_back(row-1, col-1))
    return cellids

def _df2cell_time_array(data):
    # reshape dataframe to ndarray of size (n_timesteps, n_cells)
    data = data.reset_index()
    data = data.pivot(index='time', columns='cellid', values='internet')
    data = data.fillna(0)
    data = data.values
    # print("reshaped data to shape {}".format(data.shape))
    return data

def _get_grids_by_cellids(data: pd.DataFrame, cellids: list) -> pd.DataFrame:
    """
    Get the grids of the given cellids
    :param data: the dataframe
    :param cellids: the cellids
    :return: the data of cell ids
    """
    return data.loc[data['cellid'].isin(cellids)]

def _filter_grids_data_by_colrow(data: pd.DataFrame, grid_range) -> pd.DataFrame:
    """
    Get data of grids by the given col and row
    :param data: the dataframe
    :param grid_range: the range of grids, (row_min, row_max, col_min, col_max)
    """
    cellids = _gen_cellids_by_colrow(grid_range)
    data = _get_grids_by_cellids(data, cellids)
    data = data.sort_values(['cellid', 'time']).reset_index(drop=True)
    return data

class Milan():
    def __init__(self,
                 data_dir: str = 'data/sms-call-internet-mi',
                 grid_range: tuple = (41, 70, 41, 70),
                 aggr_time: str = None,
                 out_len: int = 1,
                 batch_size: int = 64,
                 normalize: bool = False,
                 max_norm: float = 1.,
                 tele_column: str = 'internet',
                 time_range: str = 'all',
                 ):
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.aggr_time = aggr_time

        if tele_column not in ['internet', 'smsin', 'smsout', 'callin', 'callout']:
            raise ValueError('tele_column must be one of internet, smsin, smsout, callin, callout')
        self.tele_column = tele_column

        self.time_range = time_range
        if time_range not in ['30days', 'all']:
            raise ValueError('time_range must be one of 30days, all')
        # milan_internet_all_data.csv.gz | milan_telecom_data.csv.gz
        self.data_dir = data_dir
        self.file_name = "milan_internet_all_data.csv.gz" if time_range == 'all' else "milan_telecom_data.csv.gz"
        self.val_split_date = self.get_default_split_date(time_range)['val']
        self.test_split_date = self.get_default_split_date(time_range)['test']

        self.grid_range = grid_range # (min_row, max_row, min_col, max_col)
        self.n_rows = grid_range[1] - grid_range[0] + 1
        self.n_cols = grid_range[3] - grid_range[2] + 1
        self.n_grids = self.n_rows * self.n_cols

        self.normalize = normalize
        self.max_norm = max_norm
        self.scaler = None
        self.batch_size = batch_size
        self.out_len = out_len

        self.milan_train = None
        self.milan_val = None
        self.milan_test = None

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
    def get_default_split_date(time_range) -> dict:
        # return val and test split date
        if time_range == '30days':
            return {'val': {'year': 2013, 'month': 11, 'day': 17}, 
                    'test': {'year': 2013, 'month': 11, 'day': 21}}
        else:
            return {'val': {'year': 2013, 'month': 12, 'day': 18}, 
                    'test': {'year': 2013, 'month': 12, 'day': 22}}

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            raise FileNotFoundError("{} not found".format(self.file_name))
            # start_date = self.dataset_start_date['day']
            # end_date = self.dataset_end_date['day']
            # if end_date > 10:
            #     paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) 
            #              for i in range(start_date, 10)] + \
            #             ['sms-call-internet-mi-2013-11-{i}.csv'.format(i=i) 
            #              for i in range(10, end_date)]
            # else:
            #     paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) 
            #              for i in range(start_date, end_date)]
            # paths = [os.path.join(self.data_dir, path) for path in paths]
            # load_and_save_telecom_data_by_tele(paths, self.data_dir, tele_column=self.tele_column)
        else:
            print('{} already exists in {}'.format(self.file_name, self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            raise FileNotFoundError('{} not found in {}'.format(self.file_name, self.data_dir))
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_len, val_len = self._load_data()
            milan_train, milan_val, milan_test = Milan.train_test_split(self.milan_grid_data, train_len, val_len)
            self.milan_train = milan_train.reshape(-1, self.n_rows, self.n_cols)
            self.milan_val = milan_val.reshape(-1, self.n_rows, self.n_cols)
            self.milan_test = milan_test.reshape(-1, self.n_rows, self.n_cols)
            print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))
        # Assign test dataset for use in dataloader(s)
        if stage in ["test", "predict"] or stage is None:
            if self.milan_test is None:
                train_len, val_len = self._load_data()
                milan_train, milan_val, milan_test = Milan.train_test_split(self.milan_grid_data, train_len, val_len)
                self.milan_test = milan_test.reshape(-1, self.n_rows, self.n_cols)
                print('test shape: {}'.format(self.milan_test.shape))

    def _load_data(self):
        print('Loading Milan data...')
        if self.aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        filePath = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(filePath):
            raise FileNotFoundError("file {} not found".format(filePath))

        milan_data = pd.read_csv(filePath, compression='gzip', usecols=['cellid', 'time', 'internet'])
        milan_data['time'] = pd.to_datetime(milan_data['time'], format='%Y-%m-%d %H:%M:%S')
        milan_data = _filter_grids_data_by_colrow(milan_data, self.grid_range)
        if self.aggr_time == 'hour':
            milan_data = milan_data.groupby(['cellid', pd.Grouper(key="time", freq="1H")]).sum()
            milan_data.reset_index(inplace=True)
        self.milan_df = milan_data

        # reshape dataframe to ndarray of size (n_timesteps, n_cells)
        milan_grid_data = _df2cell_time_array(milan_data)
        if self.normalize:
            self.scaler = MinMaxScaler((0, self.max_norm))
            milan_grid_data = self.scaler.fit_transform(milan_grid_data.reshape(-1, 1)).reshape(milan_grid_data.shape)
            # milan_grid_data = scaler.fit_transform(milan_grid_data)
        # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
        self.milan_grid_data = milan_grid_data.astype(np.float32)
        print("loaded {} rows and {} grids".format(milan_grid_data.shape[0], milan_grid_data.shape[1]))

        train_len = self.milan_df['time'][self.milan_df['time'] < pd.Timestamp(
            **self.val_split_date)].unique().shape[0]
        val_len = self.milan_df['time'][(self.milan_df['time'] >= pd.Timestamp(
            **self.val_split_date)) & (self.milan_df['time'] < pd.Timestamp(**self.test_split_date))].unique().shape[0]
        return train_len, val_len
