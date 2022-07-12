import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from networkx import adjacency_matrix
from networkx.generators import grid_2d_graph
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from utils.milano_grid import gen_cellids_by_colrow


class Milan(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/sms-call-internet-mi',
                 grid_range: tuple = (31, 60, 41, 70),
                 aggr_time: str = None,
                 pred_len: int = 1,
                 batch_size: int = 64,
                 normalize: bool = False,
                 max_norm: float = 1.,
                 tele_column: str = 'internet',
                 time_range: str = 'all',
                 compare_mvstgn: bool = False,
                 ):
        super(Milan, self).__init__()
        self.compare_mvstgn = compare_mvstgn

        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.aggr_time = aggr_time

        if tele_column not in ['internet', 'mobile', 'call', 'callin', 'callout', 'sms', 'smsin', 'smsout']:
            raise ValueError('tele_column must be one of internet, mobile')
        self.tele_column = tele_column

        self.time_range = time_range
        if time_range not in ['30days', 'all']:
            raise ValueError('time_range must be one of 30days, all')
        # milan_internet_all_data.csv.gz | milan_telecom_data.csv.gz
        self.data_dir = data_dir
        self.file_name = "milan_{}_{}_data.csv.gz".format(tele_column, time_range)
        self.val_split_date = self.get_default_split_date(time_range)['val']
        self.test_split_date = self.get_default_split_date(time_range)['test']
        self.end_date = self.get_default_split_date(time_range)['end']

        self.grid_range = grid_range # (min_row, max_row, min_col, max_col)
        if grid_range is None:
            self.grid_range = (1, 100, 1, 100)
            self.n_rows = self.n_cols = 100
        else:
            self.n_rows = grid_range[1] - grid_range[0] + 1
            self.n_cols = grid_range[3] - grid_range[2] + 1
        self.n_grids = self.n_rows * self.n_cols
        self.adj_mx = adjacency_matrix(grid_2d_graph(self.n_rows, self.n_cols))

        self.normalize = normalize
        self.max_norm = max_norm
        self.scaler = None
        self.batch_size = batch_size
        self.pred_len = pred_len

        self.milan_train = None
        self.milan_val = None
        self.milan_test = None

    @staticmethod
    def train_test_split(data: np.ndarray, train_len: int, val_len: int=None, test_len: int=None, is_val=True) -> tuple:
        train = data[:train_len, :]
        if is_val:
            val = data[train_len:train_len+val_len, :]
            test = data[train_len+val_len:train_len+val_len+test_len, :]
            return train, val, test
        else:
            test = data[train_len:, :]
            return train, test

    @staticmethod
    def get_default_split_date(time_range) -> dict:
        # return val and test split date
        if time_range == '30days':
            return {'val': {'year': 2013, 'month': 11, 'day': 18}, 
                    'test': {'year': 2013, 'month': 11, 'day': 21},
                    'end': {'year': 2013, 'month': 12, 'day': 1}}
        else:
            return {'val': {'year': 2013, 'month': 12, 'day': 19},  # 10 | 13 | 19
                    'test': {'year': 2013, 'month': 12, 'day': 26}, # 14 | 23 | 26
                    'end': {'year': 2014, 'month': 1, 'day': 2}}    # 24 |  2 |  2

    @staticmethod
    def _load_telecom_data(path):
        print("loading data from file: {}".format(path))
        data = pd.read_csv(path, header=0, index_col=0)
        # test TODO DEBUG
        # data = data[data['countrycode'] == 39]
        data = data.groupby(['cellid', 'time'], as_index=False).sum()
        data.drop(['countrycode'], axis=1, inplace=True)
        return data

    def prepare_data(self):
        if self.compare_mvstgn: 
            return # use data_git_version.h5

        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            # raise FileNotFoundError("{} not found".format(self.file_name))
            enddate = self.get_default_split_date(self.time_range)['end']
            # paths = ['sms-call-internet-mi-2013-11-01.csv']
            paths = ['sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) 
                        for i in range(1, 10)] + \
                    ['sms-call-internet-mi-2013-11-{i}.csv'.format(i=i) 
                        for i in range(10, 30+1)]
            if enddate['month'] == 1:
                paths += ['sms-call-internet-mi-2013-12-0{i}.csv'.format(i=i) 
                            for i in range(1, 10)] + \
                        ['sms-call-internet-mi-2013-12-{i}.csv'.format(i=i) 
                            for i in range(10, 31+1)]
                paths += ['sms-call-internet-mi-2014-01-01.csv']
            
            paths = [os.path.join(self.data_dir, path) for path in paths]

            data = pd.DataFrame()
            for path in paths:
                data = pd.concat([data, self._load_telecom_data(path)], ignore_index=True)
            data = data.sort_values(['cellid', 'time']).reset_index(drop=True)
            print("loaded {} rows".format(len(data)))
            if self.tele_column not in ['internet', 'callin', 'callout', 'smsin', 'smsout', 'mobile', 'sms', 'call']:
                raise ValueError("tele_column must be one of internet, callin, callout, smsin, smsout, mobile, sms, call")
            if self.tele_column == 'mobile':
                data['mobile'] = data['smsin'] + data['smsout'] + data['callin'] + data['callout'] + data['internet']
            elif self.tele_column == 'call':
                data['call'] = data['callin'] + data['callout']
            elif self.tele_column == 'sms':
                data['sms'] = data['smsin'] + data['smsout']
            data = data[['cellid', 'time', self.tele_column]]
            data.to_csv(os.path.join(
                      self.data_dir, 'milan_{}_{}_data.csv.gz'.format(self.tele_column, self.time_range)), compression='gzip', index=False)
        else:
            print('{} already exists in {}'.format(self.file_name, self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.compare_mvstgn:
            path = os.path.join(self.data_dir, 'data_git_version.h5')
            if not os.path.exists(path):
                raise FileNotFoundError("{} not found".format(path))
            f = h5py.File(path, 'r')
            if self.tele_column == 'sms':
                data = f['data'][:, :, 0]
            elif self.tele_column == 'call':
                data = f['data'][:, :, 1]
            elif self.tele_column == 'internet':
                data = f['data'][:, :, 2]
            else:
                raise ValueError("{} is not a valid column".format(self.tele_column))
            timesampes = pd.to_datetime(f['idx'][:].astype(str), format='%Y-%m-%d %H:%M:%S')

            if self.normalize:
                self.scaler = MinMaxScaler((0, self.max_norm))
                data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

            val_len = test_len = 168
            train_len = len(data) - val_len - test_len
            self.milan_timestamps = {
                "train": timesampes[:train_len],
                "val": timesampes[train_len:train_len+val_len],
                "test": timesampes[train_len+val_len:train_len+val_len+test_len],
            }
            data = data.reshape(-1, 100, 100)
            # crop data by grid_range
            data = data[:, self.grid_range[0]-1:self.grid_range[1], self.grid_range[2]-1:self.grid_range[3]]

            self.milan_train, self.milan_val, self.milan_test = self.train_test_split(data, train_len, val_len, test_len)
            print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))
            return

        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            raise FileNotFoundError('{} not found in {}'.format(self.file_name, self.data_dir))
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_len, val_len, test_len = self._load_data()
            self.milan_timestamps = {
                "train": self.milan_df['time'].iloc[:train_len],
                "val": self.milan_df['time'].iloc[train_len:train_len+val_len],
                "test": self.milan_df['time'].iloc[train_len+val_len:train_len+val_len+test_len],
            }
            milan_train, milan_val, milan_test = Milan.train_test_split(self.milan_grid_data, train_len, val_len, test_len)
            self.milan_train = milan_train.reshape(-1, self.n_rows, self.n_cols)
            self.milan_val = milan_val.reshape(-1, self.n_rows, self.n_cols)
            self.milan_test = milan_test.reshape(-1, self.n_rows, self.n_cols)
            print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))
        # Assign test dataset for use in dataloader(s)
        if stage in ["test", "predict"] or stage is None:
            if self.milan_test is None:
                train_len, val_len, test_len = self._load_data()
                self.milan_timestamps = {
                    "train": self.milan_df['time'].iloc[:train_len],
                    "val": self.milan_df['time'].iloc[train_len:train_len+val_len],
                    "test": self.milan_df['time'].iloc[train_len+val_len:train_len+val_len+test_len],
                }
                milan_train, milan_val, milan_test = Milan.train_test_split(self.milan_grid_data, train_len, val_len, test_len)
                self.milan_test = milan_test.reshape(-1, self.n_rows, self.n_cols)
                print('test shape: {}'.format(self.milan_test.shape))

    def _load_data(self):
        print('Loading Milan data...')
        if self.aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        filePath = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(filePath):
            raise FileNotFoundError("file {} not found".format(filePath))

        milan_data = pd.read_csv(filePath, compression='gzip', usecols=['cellid', 'time', self.tele_column])
        milan_data['time'] = pd.to_datetime(milan_data['time'], format='%Y-%m-%d %H:%M:%S')
        if self.grid_range is not None:
            cellids = gen_cellids_by_colrow(self.grid_range)
            milan_data = milan_data.loc[milan_data['cellid'].isin(cellids)]
        milan_data = milan_data.sort_values(['cellid', 'time']).reset_index(drop=True)
        if self.aggr_time == 'hour':
            milan_data = milan_data.groupby(['cellid', pd.Grouper(key="time", freq="1H")]).sum()
            milan_data.reset_index(inplace=True)
        self.milan_df = milan_data

        # reshape dataframe to ndarray of size (n_timesteps, n_cells)
        milan_grid_data = milan_data.pivot(index='time', columns='cellid', values=self.tele_column)
        milan_grid_data = milan_grid_data.replace([np.inf, -np.inf], np.nan)
        milan_grid_data = milan_grid_data.fillna(0).values
        # for debug -> set max to 2000
        # milan_grid_data = milan_grid_data.clip(max=1000.)

        if self.normalize:
            self.scaler = MinMaxScaler((0, self.max_norm))
            milan_grid_data = self.scaler.fit_transform(milan_grid_data.reshape(-1, 1)).reshape(milan_grid_data.shape)
        # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
        self.milan_grid_data = milan_grid_data.astype(np.float32)
        print("loaded {} rows and {} grids".format(milan_grid_data.shape[0], milan_grid_data.shape[1]))

        train_len = self.milan_df['time'][self.milan_df['time'] < pd.Timestamp(
            **self.val_split_date)].unique().shape[0]
        val_len = self.milan_df['time'][(self.milan_df['time'] >= pd.Timestamp(
            **self.val_split_date)) & (self.milan_df['time'] < pd.Timestamp(**self.test_split_date))].unique().shape[0]
        test_len = self.milan_df['time'][(self.milan_df['time'] >= pd.Timestamp(
            **self.test_split_date)) & (self.milan_df['time'] < pd.Timestamp(**self.end_date))].unique().shape[0]

        return train_len, val_len, test_len

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError


# test
if __name__ == '__main__':
    milan = Milan(time_range='all', aggr_time='hour', tele_column='sms', grid_range=None)
    milan.prepare_data()
    milan.setup()   
    sms_milan_train = np.concatenate((milan.milan_train, milan.milan_val), axis=0)
    sms_milan_test = milan.milan_test
    print(sms_milan_train[0][0])
