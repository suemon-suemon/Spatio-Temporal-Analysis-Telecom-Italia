import os
from typing import Optional
import json
import subprocess
import h5py
import numpy as np
import pandas as pd
from networkx import adjacency_matrix
from networkx.generators import grid_2d_graph
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler


class Milan(LightningDataModule):
    def __init__(self,
                 data_dir: str = '/data/scratch/jiayin',
                 download_json_name: str = 'Milan_telecom_urls.json',
                 grid_range: tuple = (41, 60, 41, 60),
                 aggr_time = None,
                 pred_len: int = 1,
                 batch_size: int = 64,
                 normalize: bool = True,
                 tele_column: str = 'internet',
                 time_range: str = 'all',
                 compare_mvstgn: bool = False,
                 load_meta: bool = True,
                 impute_missing: bool = True,
                 ):
        super(Milan, self).__init__()
        self.compare_mvstgn = compare_mvstgn
        self.load_meta = load_meta
        self.meta_file_name = 'crawled_feature.csv'
        self.meta = None  # 默认初始化 meta 属性
        self.impute_missing = impute_missing
        self.download_json_name = download_json_name
        self.aggr_time = aggr_time
        self.N_all = (grid_range[1] - grid_range[0] + 1) * (grid_range[3] - grid_range[2] + 1)
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        if tele_column not in ['internet', 'call', 'callin', 'callout', 'sms', 'smsin', 'smsout', 'sms2', 'call2']:
            raise ValueError('tele_column must be one of internet, call, callin, callout, sms, smsin, smsout')
        self.tele_column = tele_column

        self.time_range = time_range
        if time_range not in ['30days', 'all']:
            raise ValueError('time_range must be one of 30days, all')
        # milan_internet_all_data.csv.gz | milan_telecom_data.csv.gz
        self.data_dir = data_dir
        self.file_name = "milan_{}_T_N_5.h5".format(
            '10min' if self.aggr_time is None else self.aggr_time)

        self.grid_range = grid_range  # (min_row, max_row, min_col, max_col)
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
    def train_test_split(data: np.ndarray, train_len: int, val_len: int = None, test_len: int = None,
                         is_val=True) -> tuple:
        train = data[:train_len, :]
        if is_val:
            val = data[train_len:train_len + val_len, :]
            test = data[train_len + val_len:train_len + val_len + test_len, :]
            return train, val, test
        else:
            test = data[train_len:, :]
            return train, test

    def get_default_len(self) -> tuple:
        # 计算训练集长度（向下取整）
        train_len = int(self.T * self.train_ratio)
        # 计算验证集长度（向下取整）
        val_len = int(self.T * self.val_ratio)
        # 测试集长度为剩余部分，确保总长度一致
        test_len = self.T - train_len - val_len
        return train_len, val_len, test_len

    # @staticmethod
    # def get_default_len(time_range) -> dict:
    #     # return train, val and test length
    #     if time_range == '30days':
    #         tvt = (16, 7, 7)
    #         return (i * 24 * 6 for i in tvt)
    #         # return {'val': {'year': 2013, 'month': 11, 'day': 18},
    #         #         'test': {'year': 2013, 'month': 11, 'day': 21},
    #         #         'end': {'year': 2013, 'month': 12, 'day': 1}}
    #     else:
    #         tvt = (48, 7, 7)
    #         return (i * 24 for i in tvt)
    #         # return {'val': {'year': 2013, 'month': 12, 'day': 16},  # 10 | 13 | 19
    #         #         'test': {'year': 2013, 'month': 12, 'day': 23}, # 14 | 23 | 26
    #         #         'end': {'year': 2014, 'month': 1, 'day': 2}}    # 24 |  2 |  2

    @staticmethod
    def _load_telecom_data(path):
        print("loading data from file: {}".format(path))
        data = pd.read_csv(path, header=0, index_col=0)
        data.reset_index(inplace=True)  # 重置索引，将 'cellid' 恢复为普通列
        # 对数据根据 ['cellid', 'time'] 进行分组并求和
        # 这一步会合并同一传感器在同一时间点的多条记录
        data = data.groupby(['cellid', 'time'], as_index=False).sum()
        data.drop(['countrycode'], axis=1, inplace=True)
        return data

    # original version：
    # @staticmethod
    # def _load_telecom_data(path):
    #     print("loading data from file: {}".format(path))
    #     data = pd.read_csv(path, header=0, index_col=0)
    #     data = data.groupby(['cellid', 'time'], as_index=False).sum()
    #     data.drop(['countrycode'], axis=1, inplace=True)
    #     return data

    def download_data(self):
        # json文件中存储了网页上扒下来的url下载链接
        # 设置保存文件的目标目录
        target_dir = self.data_dir
        download_json_name = self.download_json_name
        os.makedirs(target_dir, exist_ok=True)

        # 打开并读取 JSON 文件
        with open(os.path.join(target_dir, download_json_name), "r", encoding="utf-8") as f:
            file_list = json.load(f)

        for file_obj in file_list:
            # 提取文件名和下载链接
            file_name = file_obj.get("name")
            content_url = file_obj.get("contentUrl")

            if file_name and content_url:
                # 构造保存路径
                save_path = os.path.join(target_dir, file_name)
                # 构造 curl 命令
                cmd = ["curl", "-L", content_url, "-o", save_path]
                print(f"Downloading {file_name} ...")
                subprocess.run(cmd, check=True)
            else:
                print("缺少文件名或下载链接:", file_obj)

    def prepare_data(self):
        # 处理 meta 文件
        if self.load_meta:
            meta_path = os.path.join(self.data_dir, self.meta_file_name)
            if not os.path.exists(meta_path):  # 如果meta-path文件路径不存在
                raise FileNotFoundError("{} not found".format(self.meta_file_name))
            else:  # meta-path文件路径存在时
                print('{} already exists in {}'.format(self.meta_file_name, self.data_dir))
                meta = pd.read_csv(meta_path, header=0)
                meta = meta.values.T
                self.meta = meta.reshape(-1, 100, 100)

        if self.compare_mvstgn:
            return  # use data_git_version.h5

        # 生成milan 数据文件：milan_5min_T_N_5.h5  或 milan_5min_T_N_5.h5
        file_path = os.path.join(self.data_dir, self.file_name)
        column_names = ['cellid', 'time', 'countrycode', 'smsin', 'smsout', 'callin', 'callout', 'internet']

        if not os.path.exists(file_path):
            # 构造需要加载的 CSV 文件列表
            paths = ['sms-call-internet-mi-2013-11-{d}.csv'.format(d=str(i).zfill(2)) for i in range(1, 31)]
            paths += ['sms-call-internet-mi-2013-12-{d}.csv'.format(d=str(i).zfill(2)) for i in range(1, 32)]
            paths += ['sms-call-internet-mi-2014-01-01.csv']
            # paths = ['sms-call-internet-mi-2013-12-31.csv']
            # paths += ['sms-call-internet-mi-2014-01-01.csv']  # 测试用
            paths = [os.path.join(self.data_dir, path) for path in paths]

            # 如果 CSV 文件不存在，则将对应的 TXT 文件转换为 CSV
            for path in paths:
                if not os.path.exists(path):
                    txt_path = path.replace('.csv', '.txt')
                    if os.path.exists(txt_path):
                        print("CSV file {} not found. Converting {} to CSV format.".format(path, txt_path))
                        # 原有的 TXT 文件以制表符分隔
                        df_temp = pd.read_csv(txt_path, sep='\t', header=None, names=column_names)
                        df_temp.to_csv(path, index=False)
                    else:
                        print("Neither CSV nor TXT file found for {}".format(path))

            # 加载所有 CSV 文件的数据
            data = pd.DataFrame()
            for path in paths:
                data = pd.concat([data, self._load_telecom_data(path)], ignore_index=True)
            print("loaded {} rows".format(len(data)))
            data['time'] = pd.to_datetime(data['time'], unit='ms')  # 原时间格式为ms

            # 分组聚合
            # 利用 pd.Grouper 将数据按照每 10 分钟一个时间段进行分组，
            # 然后对每个分组（每个 cellid 在每 10 分钟内的记录）进行求和
            if self.aggr_time == 'hour':
                data = data.groupby(['cellid', pd.Grouper(key="time", freq="1H")]).sum()
            elif self.aggr_time == '10min':
                data = data.groupby(['cellid', pd.Grouper(key="time", freq="10T")]).sum()
            elif self.aggr_time is None:
                data = data.groupby(['cellid', pd.Grouper(key="time", freq="10T")]).sum()

            data.reset_index(inplace=True)

            # 将聚合后的 DataFrame 中的 ‘time’ 列去重，并转换为字符串形式
            timestamps = data['time'].drop_duplicates().dt.strftime('%Y-%m-%d %H:%M:%S').values
            data['time'] = pd.to_datetime(data['time'], unit='ms')

            # 将数据 pivot 成多指标的格式，再 reshape 为 (T, N, 5)
            data = data.pivot(index='cellid', columns='time',
                              values=['smsin', 'smsout', 'callin', 'callout', 'internet'])
            data = data.values
            data = data.reshape(10000, 5, -1).transpose(2, 0, 1)  # (T, N, 5)

            assert len(timestamps) == data.shape[0]
            # 保存为 HDF5 文件
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=data)
                f.create_dataset('time', data=timestamps)
            print("saved to {}".format(file_path))
        else: # 存在 milan_10min_T_N_5.h5
            print('{} already exists in {}'.format(self.file_name, self.data_dir))

    def setup(self, stage: Optional[str] = None) -> None:
        # 如果设置为加载元数据（meta），则对 meta 数据按照网格范围进行裁剪
        if self.load_meta:
            # 裁剪 meta 数据，使其只包含 grid_range 指定的区域
            self.meta = self.meta[:, self.grid_range[0] - 1:self.grid_range[1],
                        self.grid_range[2] - 1:self.grid_range[3]]
            print("loaded meta of shape: {} ".format(self.meta.shape))

        # 如果使用 compare_mvstgn 模式（用于比较不同模型版本的数据），加载特定的 HDF5 文件
        if self.compare_mvstgn:
            # 构造 git 版本数据文件的完整路径
            path = os.path.join(self.data_dir, 'data_git_version.h5')
            # 如果文件不存在则抛出错误
            if not os.path.exists(path):
                raise FileNotFoundError("{} not found".format(path))
            # 打开 HDF5 文件，模式为只读
            f = h5py.File(path, 'r')
            # 根据 tele_column 参数选择对应的电信数据
            if self.tele_column == 'sms':
                data = f['data'][:, :, 0]
            elif self.tele_column == 'call':
                data = f['data'][:, :, 1]
            elif self.tele_column == 'internet':
                data = f['data'][:, :, 2]
            else:
                raise ValueError("{} is not a valid column".format(self.tele_column))
            # 读取索引（通常是时间戳）并转换为 datetime 格式，格式为 'YYYY-MM-DD HH:MM:SS'
            self.timesampes = pd.to_datetime(f['idx'][:].astype(str), format='%Y-%m-%d %H:%M:%S')

            # 如果设置了数据归一化，则利用 MinMaxScaler 将数据缩放到 [0, 1] 范围内
            if self.normalize:
                self.scaler = MinMaxScaler((0, 1))
                data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

            # 设置验证集和测试集的长度，均为 168（可以理解为 168 小时或 168 个时间步）
            val_len = test_len = 168
            # 训练集长度为总长度减去验证集和测试集的长度
            train_len = len(data) - val_len - test_len
            # 将数据 reshape 为 (T, 100, 100) 的形状，其中 T 表示时间步数，100×100 表示原始网格尺寸
            data = data.reshape(-1, 100, 100)
            # 根据 grid_range 裁剪数据，只保留指定的网格区域（注意 Python 索引从0开始）
            data = data[:, self.grid_range[0] - 1:self.grid_range[1], self.grid_range[2] - 1:self.grid_range[3]]
            # 此处直接 return，不再继续执行后面的加载数据流程
            return

        # 如果 compare_mvstgn 为 False，则检查预期的 HDF5 数据文件是否存在
        if not os.path.exists(os.path.join(self.data_dir, self.file_name)):
            raise FileNotFoundError('{} not found in {}'.format(self.file_name, self.data_dir))
        # 调用内部方法 _load_data() 加载数据，并为后续的 dataloader 分配训练/验证/测试数据
        self._load_data()

    def _load_data(self):
        if hasattr(self, 'milan_grid_data'):
            return

        print('Loading Milan data...')
        filePath = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(filePath):
            raise FileNotFoundError("file {} not found".format(filePath))

        with h5py.File(filePath, 'r') as f:
            data = f['data'][:]  # shape (T_len, N_grids, 5)
            self.T = data.shape[0] # 1413(hour) or 8496(10min)
            self.timestamps = pd.to_datetime(f['time'][:].astype(str), format='%Y-%m-%d %H:%M:%S')

        if self.tele_column == 'smsin':
            data = data[:, :, 0:1]
        elif self.tele_column == 'smsout':
            data = data[:, :, 1:2]
        elif self.tele_column == 'callin':
            data = data[:, :, 2:3]
        elif self.tele_column == 'callout':
            data = data[:, :, 3:4]
        elif self.tele_column == 'internet':
            data = data[:, :, 4:5]
        elif self.tele_column == 'sms':
            data = data[:, :, 0:1] + data[:, :, 1:2]
        elif self.tele_column == 'call':
            data = data[:, :, 2:3] + data[:, :, 3:4]
        elif self.tele_column == 'sms2':
            data = data[:, :, 0:2]
        elif self.tele_column == 'call2':
            data = data[:, :, 2:4]
        else:
            raise ValueError("{} is not a valid column".format(self.tele_column))

        data = data.reshape(data.shape[0], 100, 100, -1).transpose((0, 3, 1, 2))
        if self.grid_range is not None:
            oridata = data
            data = data[:, :, self.grid_range[0] - 1:self.grid_range[1], self.grid_range[2] - 1:self.grid_range[3]]

        for id in np.argwhere(np.isnan(data)):
            if self.impute_missing:
                oriid = [id[0], id[1], id[2] + self.grid_range[0] - 1, id[3] + self.grid_range[2] - 1]
                surroundings = np.array([oridata[oriid[0], oriid[1], oriid[1] - 1, oriid[2]],
                                         oridata[oriid[0], oriid[1], oriid[1] + 1, oriid[2]],
                                         oridata[oriid[0], oriid[1], oriid[1], oriid[2] - 1],
                                         oridata[oriid[0], oriid[1], oriid[1], oriid[2] + 1]])
                data[id[0], id[1], id[2], id[3]] = np.nanmean(surroundings)
            else:
                data[id[0], id[1], id[2], id[3]] = 0
        if self.normalize:
            self.scaler = MinMaxScaler((0, 1))
            data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

        # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
        self.milan_grid_data = data.astype(np.float32)
        print("loaded data shape: ", data.shape)

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError

def get_indexes_of_train(format, time_level, out_start_idx, close_len, period_len, trend_len=0, *, pred_len=1):
    """
    根据给定的时间格式、时间级别等信息生成训练数据的索引。

    参数：
    format (str): 数据格式类型，决定生成索引的方式。可选值为 'default' 和 'sttran'。
    time_level (str): 时间级别，决定一天中的时间步数。可以是 'hour'（每小时记录）或 '10 mins'（每10分钟记录）。
    out_start_idx (int): 输出开始的时间步索引。
    close_len (int): 用于回顾的历史时间步数（包括当前时间步）。
    period_len (int): 用于捕获周期性变化的时间段长度（单位为天）。
    trend_len (int, optional): 用于捕获长期趋势的时间段长度（单位为周），默认为0（不使用）。
    pred_len (int, optional): 预测的时间步数，默认为1。

    返回：
    indices (list): 训练数据的索引列表。

    功能：
    - 根据不同的时间级别（小时或每10分钟）计算一天的时间步数（24小时或24小时*6）。
    - 根据数据格式 'default' 或 'sttran' 生成训练数据的索引。'default' 格式会生成历史数据索引并加上周期性和趋势性的数据，'sttran' 格式主要用于时序变换模型，生成带有历史时间步的索引。
    """
    # 定义一天的时间步数，基于时间级别决定是24小时还是24*6个10分钟段
    if time_level == 'hour':
        TIME_STEPS_OF_DAY = 24
    else:  # 如果是每10分钟一条记录，则一天有24 * 6个时间步
        TIME_STEPS_OF_DAY = 24 * 6

    indices = []  # 初始化索引列表

    # 'default'格式：使用历史数据以及周期性和趋势性的数据来生成索引
    if format == 'default':
        # 首先加入从当前时间步开始，回溯close_len个时间步的数据
        indices += [out_start_idx - i - 1 for i in range(close_len)]

        # 如果period_len大于0，加入周期性数据的索引（跨越多个天）
        if period_len > 0:
            indices += [
                out_start_idx - (i + 1) * TIME_STEPS_OF_DAY + (pred_len - j - 1)
                for j in range(pred_len) for i in range(period_len)
            ]

        # 如果trend_len大于0，加入趋势性数据的索引（跨越多个星期）
        if trend_len > 0:
            indices += [
                out_start_idx - (i + 1) * TIME_STEPS_OF_DAY * 7 + (pred_len - j - 1)
                for j in range(pred_len) for i in range(trend_len)
            ]

    # 'sttran'格式：生成包含历史时间步的索引，适用于时序变换模型
    elif format == 'sttran':
        if period_len > 0:
            indices += [
                out_start_idx - (i + 1) * TIME_STEPS_OF_DAY - j
                for j in range(close_len) for i in range(period_len)
            ]

    # 将索引顺序反转
    indices.reverse()

    return indices

# test
if __name__ == '__main__':
    milan = Milan(data_dir='/data/scratch/jiayin',
                  download_json_name='Milan_telecom_urls.json',
                  aggr_time='10min',
                  time_range='all',
                  tele_column='internet',
                  grid_range=(41, 70, 41, 70),
                  load_meta=True)
    # milan.download_data()
    milan.prepare_data()
    milan.setup()

    # sms_milan_train = np.concatenate((milan.milan_train, milan.milan_val), axis=0)
    # sms_milan_test = milan.milan_test
    # print('sms_milan_train: shape', sms_milan_train.shape)
    # print('sms_milan_test: shape', sms_milan_test.shape)
