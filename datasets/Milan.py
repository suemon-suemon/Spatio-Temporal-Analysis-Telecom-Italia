import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from networkx import adjacency_matrix
from networkx.generators import grid_2d_graph
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler


class Milan(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/sms-call-internet-mi',
                 grid_range: tuple = (41, 60, 41, 60),
                 aggr_time: str = None,
                 pred_len: int = 1,
                 batch_size: int = 64,
                 normalize: bool = False,
                 tele_column: str = 'internet',
                 time_range: str = 'all',
                 compare_mvstgn: bool = False,
                 load_meta: bool = True,
                 ): 
        super(Milan, self).__init__()
        self.compare_mvstgn = compare_mvstgn
        self.load_meta = load_meta
        self.meta_file_name = 'crawled_feature.csv'

        if aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        self.aggr_time = aggr_time

        if tele_column not in ['internet', 'call', 'callin', 'callout', 'sms', 'smsin', 'smsout']:
            raise ValueError('tele_column must be one of internet, call, callin, callout, sms, smsin, smsout')
        self.tele_column = tele_column

        self.time_range = time_range
        if time_range not in ['30days', 'all']:
            raise ValueError('time_range must be one of 30days, all')
        # milan_internet_all_data.csv.gz | milan_telecom_data.csv.gz
        self.data_dir = data_dir
        self.file_name = "milan_{}.h5".format('hr' if self.aggr_time == 'hour' else 'min')

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
    def get_default_len(time_range) -> dict:
        # return train, val and test length
        if time_range == '30days':
            tvt = (16, 7, 7)
            return (i*24*6 for i in tvt)
            # return {'val': {'year': 2013, 'month': 11, 'day': 18}, 
            #         'test': {'year': 2013, 'month': 11, 'day': 21},
            #         'end': {'year': 2013, 'month': 12, 'day': 1}}
        else:
            tvt = (48, 7, 7)
            return (i*24 for i in tvt)
            # return {'val': {'year': 2013, 'month': 12, 'day': 16},  # 10 | 13 | 19
            #         'test': {'year': 2013, 'month': 12, 'day': 23}, # 14 | 23 | 26
            #         'end': {'year': 2014, 'month': 1, 'day': 2}}    # 24 |  2 |  2

    @staticmethod
    def _load_telecom_data(path):
        print("loading data from file: {}".format(path))
        data = pd.read_csv(path, header=0, index_col=0)
        data = data.groupby(['cellid', 'time'], as_index=False).sum()
        data.drop(['countrycode'], axis=1, inplace=True)
        return data

    def prepare_data(self):
        if self.load_meta:
            if not os.path.exists(os.path.join(self.data_dir, self.meta_file_name)):
                raise FileNotFoundError("{} not found".format(self.meta_file_name))
            else:
                print('{} already exists in {}'.format(self.meta_file_name, self.data_dir))
                meta = pd.read_csv(os.path.join(self.data_dir, self.meta_file_name), header=0)
                meta = meta.values.T
                self.meta = meta.reshape(-1, 100, 100)

        if self.compare_mvstgn: 
            return # use data_git_version.h5

        # there should be two files: milan_hr.h5 and milan_min.h5
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            # raise FileNotFoundError("{} not found".format(self.file_name))
            paths = ['sms-call-internet-mi-2013-11-{d}.csv'.format(d=str(i).zfill(2)) for i in range(1, 31)]
            paths += ['sms-call-internet-mi-2013-12-{d}.csv'.format(d=str(i).zfill(2)) for i in range(1, 32)]
            paths += ['sms-call-internet-mi-2014-01-01.csv']
            # paths = ['sms-call-internet-mi-2013-11-01.csv']
            paths = [os.path.join(self.data_dir, path) for path in paths]

            data = pd.DataFrame()
            for path in paths:
                data = pd.concat([data, self._load_telecom_data(path)], ignore_index=True)
            print("loaded {} rows".format(len(data)))
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            # data = data.sort_values(['cellid', 'time']).reset_index(drop=True)
            if self.aggr_time == 'hour':
                data = data.groupby(['cellid', pd.Grouper(key="time", freq="1H")]).sum()
                data.reset_index(inplace=True)
            timestamps = data['time'].drop_duplicates().dt.strftime('%Y-%m-%d %H:%M:%S').values
            
            data = data.pivot(index='cellid', columns='time', values=['smsin', 'smsout', 'callin', 'callout', 'internet'])
            data = data.values
            data = data.reshape(10000, 5, -1).transpose(2, 0, 1) # (T, N, 5)

            assert len(timestamps) == data.shape[0]
            # save to h5
            with h5py.File(os.path.join(self.data_dir, self.file_name), 'w') as f:
                f.create_dataset('data', data=data)
                f.create_dataset('time', data=timestamps)
            print("saved to {}".format(os.path.join(self.data_dir, self.file_name)))
        else:
            print('{} already exists in {}'.format(self.file_name, self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.load_meta:
            self.meta = self.meta[:, self.grid_range[0]-1:self.grid_range[1], self.grid_range[2]-1:self.grid_range[3]]
            print("loaded meta of shape: {} ".format(self.meta.shape))

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
                self.scaler = MinMaxScaler((0, 1))
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
        train_len, val_len, test_len = self.get_default_len(self.time_range)
        self._load_data()
        self.milan_timestamps = {
            "train": self.timestamps[:train_len],
            "val": self.timestamps[train_len:train_len+val_len],
            "test": self.timestamps[train_len+val_len:train_len+val_len+test_len],
        }
        milan_train, milan_val, milan_test = Milan.train_test_split(self.milan_grid_data, train_len, val_len, test_len)
        self.milan_train = milan_train.reshape(-1, self.n_rows, self.n_cols)
        self.milan_val = milan_val.reshape(-1, self.n_rows, self.n_cols)
        self.milan_test = milan_test.reshape(-1, self.n_rows, self.n_cols)
        print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))


    def _load_data(self):
        if hasattr(self, 'milan_grid_data'):
            return

        print('Loading Milan data...')
        if self.aggr_time not in [None, 'hour']:
            raise ValueError("aggre_time must be None or 'hour'")
        filePath = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(filePath):
            raise FileNotFoundError("file {} not found".format(filePath))

        with h5py.File(filePath, 'r') as f: # shape (T_len, N_grids, 5)
            data = f['data'][:]
            self.timestamps =pd.to_datetime(f['time'][:].astype(str), format='%Y-%m-%d %H:%M:%S')
        
        if self.tele_column == 'smsin':
            data = data[:, :, 0]
        elif self.tele_column == 'smsout':
            data = data[:, :, 1]
        elif self.tele_column == 'callin':
            data = data[:, :, 2]
        elif self.tele_column == 'callout':
            data = data[:, :, 3]
        elif self.tele_column == 'internet':
            data = data[:, :, 4]
        elif self.tele_column == 'sms':
            data = data[:, :, 0] + data[:, :, 1]
        elif self.tele_column == 'call':
            data = data[:, :, 2] + data[:, :, 3]
        else:
            raise ValueError("{} is not a valid column".format(self.tele_column))

        data = data.reshape(data.shape[0], 100, 100)
        if self.grid_range is not None:
            data = data[:, self.grid_range[0]-1:self.grid_range[1], self.grid_range[2]-1:self.grid_range[3]]

        if self.normalize:
            self.scaler = MinMaxScaler((0, 1))
            data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
        # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
        self.milan_grid_data = data.astype(np.float32)
        
        print("loaded {} rows and {} grids".format(data.shape[0], data.shape[1]))

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
    milan = Milan(time_range='all', aggr_time=None, tele_column='sms', grid_range=None, load_meta=True)
    milan.prepare_data()
    milan.setup()
    sms_milan_train = np.concatenate((milan.milan_train, milan.milan_val), axis=0)
    sms_milan_test = milan.milan_test
    print(sms_milan_train[0][0])
