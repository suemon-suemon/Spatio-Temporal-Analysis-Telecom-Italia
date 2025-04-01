import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from datasets.Milan import Milan, get_indexes_of_train

from utils.time_features import time_features


class MilanSW(Milan):
    """
    Milan 数据集的滑动窗口版本，用于时空预测任务。

    参数说明：
      - format (str): 数据集格式，可选 'normal', 'informer', 'sttran', '3comp'，默认为 'normal'。
      - close_len (int): 短期历史窗口长度，用于最近时间步的输入数据。
      - period_len (int): 周期性历史窗口长度，用于捕捉周期性特征。
      - label_len (int): 标签窗口长度，用于预测时刻前的历史数据作为标签参考。
      - pred_len (int): 预测窗口长度，表示模型需要预测多少个时间步的数据。
      - window_size (int): 空间窗口大小，用于提取局部区域数据。
      - flatten (bool): 是否将空间窗口展平为一维向量，默认为 True。
      - **kwargs: 其它传递给基类 Milan 的参数，例如 data_dir, aggr_time, tele_column, time_range, compare_mvstgn, load_meta, impute_missing 等。
    """

    def __init__(self,
                 format: str = 'normal',
                 close_len: int = 3,
                 period_len: int = 3,
                 label_len: int = 12,
                 pred_len: int = 1,
                 window_size: int = 11,
                 flatten: bool = True,
                 **kwargs):
        # 通过 kwargs 传递给基类 Milan 的初始化参数
        super(MilanSW, self).__init__(**kwargs)
        if format not in ['normal', 'informer', 'sttran', '3comp']:
            raise ValueError("format must be one of 'normal', 'informer', 'sttran', '3comp'")
        self.format = format
        self.close_len = close_len  # 近期历史数据的窗口长度
        self.period_len = period_len  # 周期性历史数据的窗口长度
        self.label_len = label_len  # 标签部分的历史数据长度
        self.pred_len = pred_len  # 预测窗口的长度
        self.flatten = flatten  # 是否将空间数据展平
        self.window_size = window_size  # 空间窗口大小（用于截取局部网格区域）

    def prepare_data(self):
        # 调用父类的 prepare_data() 进行数据下载、预处理和生成 h5 文件
        Milan.prepare_data(self)

    def setup(self, stage=None):
        # 调用父类的 setup() 进行数据集分割、加载 h5 文件等工作
        Milan.setup(self, stage)
        # 根据 DataModule 内定义的 get_default_len() 方法，计算训练、验证和测试集的长度
        train_len, val_len, test_len = self.get_default_len()
        # 根据时间戳（self.timestamps）对不同数据集进行切分
        self.milan_timestamps = {
            "train": self.timestamps[:train_len],
            "val": self.timestamps[train_len:train_len + val_len],
            # 注意这里 test 的切分考虑了窗口长度，确保连续性
            "test": self.timestamps[
                    train_len + val_len - (self.close_len + self.pred_len - 1):train_len + val_len + test_len],
        }
        # 划分数据集（按时间步数进行切分），返回训练、验证、测试数据
        self.milan_train, self.milan_val, self.milan_test = self.train_test_split(self.milan_grid_data, train_len,
                                                                                  val_len, test_len)
        # 对测试集额外拼接部分验证数据，使得测试集的窗口数据连续
        self.milan_test = np.concatenate((self.milan_val[-(self.close_len + self.pred_len - 1):], self.milan_test))
        print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape,
                                                                      self.milan_test.shape))

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train'),
                          batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val'),
                          batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test'),
                          batch_size=self.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage):

        if data.shape[1] == 1:
            data = data.squeeze()
        else:
            print('More than 1 services are considered, hence no squeeze.')

        # 根据 format 参数选择不同的数据集处理方式
        if self.format == 'informer':
            dataset = MilanSWInformerDataset(data, self.milan_timestamps[stage],
                                             aggr_time=self.aggr_time, input_len=self.close_len,
                                             window_size=self.window_size, label_len=self.label_len,
                                             pred_len=self.pred_len)
        elif self.format == 'sttran':
            dataset = MilanSWStTranDataset(data, self.aggr_time, self.close_len,
                                           self.period_len, self.pred_len)
        elif self.format == '3comp':
            dataset = MilanSW3CompDataset(data, self.aggr_time, self.close_len,
                                          self.period_len, window_size=self.window_size, flatten=self.flatten)
        else:  # default, 即 'normal'
            dataset = MilanSlidingWindowDataset(data, input_len=self.close_len,
                                                window_size=self.window_size, pred_len=self.pred_len,
                                                flatten=self.flatten)
        return dataset


class MilanSlidingWindowDataset(Dataset):
    """
    基于滑动窗口方式构造 Milan 数据集。

    参数说明：
      - milan_data (pd.DataFrame): 已处理好的 Milan 网格数据，形状为 (n_timestamps, n_grid_row, n_grid_col)。
      - window_size (int): 滑动窗口的空间大小（用于截取局部区域）。
      - input_len (int): 输入数据（历史数据）的时间步数。
      - pred_len (int): 预测数据的时间步数。
      - flatten (bool): 是否将每个窗口展平为一维向量。
    """

    def __init__(self,
                 milan_data: pd.DataFrame,
                 window_size: int = 11,
                 input_len: int = 12,
                 pred_len: int = 1,
                 flatten: bool = True):
        # 保存输入数据和参数
        self.milan_data = milan_data
        self.window_size = window_size
        self.input_len = input_len
        self.flatten = flatten
        self.pred_len = pred_len
        # 计算填充大小（取窗口大小的一半）
        pad_size = window_size // 2
        # 对数据进行填充。这里假设 milan_data 的形状为 (n_timestamps, n_grid_row, n_grid_col)
        # 第一维不填充；第二和第三维在前后各填充 pad_size 个单位，填充值为常数 0
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        # 计算数据集中窗口滑动后的样本数
        return (self.milan_data.shape[0] - self.input_len - self.pred_len + 1) * \
            self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        # 根据 index 计算对应的时间步（n_slice）、行（n_row）和列（n_col）
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        # 取出对应时间窗口的数据，并从填充后的数据中获取局部窗口
        X = self.milan_data_pad[n_slice:n_slice + self.input_len,
            n_row:n_row + self.window_size,
            n_col:n_col + self.window_size]
        if self.flatten:
            # 将每个时间步的局部窗口展平为一维
            X = X.reshape((self.input_len, self.window_size * self.window_size))
        # 获取对应预测时刻的真实值（单个时间步）
        Y = self.milan_data[n_slice + self.input_len:n_slice + self.input_len + self.pred_len,
            n_row, n_col].reshape(-1)
        return (X, Y)


class MilanSW3CompDataset(Dataset):
    """
    三分量版本的 Milan 滑动窗口数据集。
    该数据集利用通信数据生成三部分信息（例如关闭窗口、周期窗口），适用于特定模型的输入要求。

    参数说明：
      - milan_data (pd.DataFrame): Milan 数据，形状为 (n_timestamps, n_grid_row, n_grid_col)。
      - aggr_time (str): 时间聚合方式，例如 'hour' 或其它。
      - close_len (int): 近期历史窗口长度。
      - period_len (int): 周期性历史窗口长度。
      - window_size (int): 空间窗口大小。
      - flatten (bool): 是否将窗口内数据展平。
    """

    def __init__(self,
                 milan_data: pd.DataFrame,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3, *,
                 window_size: int = 11,
                 flatten: bool = True):
        self.milan_data = milan_data
        self.time_level = aggr_time  # 存储时间聚合方式
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len  # 输入长度即近期历史数据长度
        self.flatten = flatten
        self.pred_len = 1  # 固定预测长度为1
        self.window_size = window_size
        pad_size = window_size // 2
        # 对空间数据进行填充
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        # 返回数据集中样本数量（基于时间步与空间网格的组合）
        return (self.milan_data.shape[0] - self.in_len) * \
            self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        # 根据索引计算时间步、行和列索引
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        out_start_idx = n_slice + self.in_len

        # 获取对应时间步的窗口数据。先通过 _get_indexes 获取输入时间窗口索引
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, self.close_len, self.period_len)
        spatial_window = (self.window_size, self.window_size)
        idx_grid_data = self.milan_data_pad[:, n_row:n_row + self.window_size, n_col:n_col + self.window_size]
        # 对每个时间步，若索引无效则用全零数组代替
        X = np.array([idx_grid_data[i] if i >= 0 else np.zeros(spatial_window) for i in indices], dtype=np.float32)
        if self.flatten:
            X = X.reshape((-1, spatial_window[0] * spatial_window[1]))
        # Y 为预测时刻的真实值
        Y = self.milan_data[out_start_idx: out_start_idx + self.pred_len, n_row, n_col]
        return (X, Y)


class MilanSWInformerDataset(Dataset):
    """
    基于Informer模型输入要求构建的 Milan 滑动窗口数据集。
    除了输入数据外，还会生成对应的时间特征。

    参数说明：
      - milan_data (pd.DataFrame): 原始 Milan 数据，形状为 (n_timestamps, n_grid_row, n_grid_col)。
      - timestamps (pd.DataFrame): 与数据对应的时间戳，用于生成时间特征。
      - aggr_time: 时间聚合方式，如 'hour' 或其他。
      - window_size (int): 滑动窗口空间大小。
      - input_len (int): 输入数据窗口长度。
      - label_len (int): 标签窗口长度（用于提供历史信息给预测）。
      - pred_len (int): 预测窗口长度，默认预测 1 个时间步。
    """

    def __init__(self,
                 milan_data: pd.DataFrame,
                 timestamps: pd.DataFrame,
                 aggr_time=None,
                 window_size: int = 11,
                 input_len: int = 12,
                 label_len: int = 12,
                 pred_len: int = 1):
        # 去除数据多余的维度（假设数据存储形式需要 squeeze）
        self.milan_data = milan_data.squeeze()
        # 生成时间特征，timeenc=1 表示使用一种编码方式，freq 根据 aggr_time 决定
        self.timestamps = time_features(timestamps, timeenc=1,
                                        freq='h' if aggr_time == 'hour' else 't')
        pad_size = window_size // 2
        self.window_size = window_size
        self.input_len = input_len
        self.label_len = label_len
        self.pred_len = pred_len
        # 对数据进行填充：仅在空间维度进行填充，不改变时间维度
        self.milan_data_pad = np.pad(self.milan_data,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
                                     'constant', constant_values=0)

    def __len__(self):
        # 计算数据集样本数 = (时间步数 - input_len - pred_len + 1) * (n_grid_row * n_grid_col)
        return (self.milan_data.shape[0] - self.input_len - self.pred_len + 1) * \
            self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        # 根据 index 计算对应的时间、空间索引
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        # X 为输入数据窗口，取 input_len 个时间步对应的空间区域
        X = self.milan_data_pad[n_slice:n_slice + self.input_len,
            n_row:n_row + self.window_size,
            n_col:n_col + self.window_size]
        # 获取对应的时间特征（X_timefeature）和标签时间特征（Y_timefeature）
        X_timefeature = self.timestamps[n_slice:n_slice + self.input_len]
        Y_timefeature = self.timestamps[
                        n_slice + self.input_len - self.label_len: n_slice + self.input_len + self.pred_len]
        # 将 X 重塑为二维数组：形状 (input_len, window_size * window_size)
        X = X.reshape((self.input_len, self.window_size * self.window_size))
        # Y 为标签数据，取出预测时刻对应的单个网格值，并转换为二维（-1, 1）
        Y = self.milan_data[n_slice + self.input_len - self.label_len: n_slice + self.input_len + self.pred_len, n_row,
            n_col].reshape(-1, 1)
        return X, Y, X_timefeature, Y_timefeature


class MilanSWStTranDataset(Dataset):
    """
    STTran 版本的 Milan 滑动窗口数据集。
    此数据集用于 STTran 模型，通过近期数据、周期性数据以及选定的空间区域（top-K grids）作为输入。

    参数说明：
      - milan_data: 原始 Milan 数据（通常为 3D 数组，形状 (n_timestamps, n_grid_row, n_grid_col)）。
      - aggr_time (str): 时间聚合方式，目前只支持 None 或 'hour'。
      - close_len (int): 近期历史数据的长度。
      - period_len (int): 周期性历史数据的长度。
      - pred_len (int): 预测窗口长度，默认预测 3 个时间步。
      - K_grids (int): 从相关性矩阵中选择的 top-K 网格数。
    """

    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3,
                 pred_len: int = 3,
                 K_grids=20):
        if aggr_time not in [None, 'hour']:
            raise ValueError("aggr_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len  # 输入数据长度与 close_len 相同
        self.pred_len = pred_len
        self.K_grids = K_grids

        self.curr_slice = -1  # 用于缓存当前切片，避免重复计算
        self.grid_topk = None

    def __len__(self):
        # 样本数量 = (n_timestamps - in_len - pred_len + 1) * (n_grid_row * n_grid_col)
        return (self.milan_data.shape[0] - self.in_len - self.pred_len + 1) * \
            self.milan_data.shape[1] * self.milan_data.shape[2]

    def __getitem__(self, index):
        # 根据索引计算时间步、行、列
        n_slice = index // (self.milan_data.shape[1] * self.milan_data.shape[2])
        n_row = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) // self.milan_data.shape[1]
        n_col = (index % (self.milan_data.shape[1] * self.milan_data.shape[2])) % self.milan_data.shape[2]
        out_start_idx = n_slice + self.in_len

        # 如果当前时间切片与上次相同，则直接使用缓存的 grid_topk
        if n_slice == self.curr_slice:
            grids_topk = self.grid_topk
        else:
            # 计算近期数据 Xc，形状 (close_len, n_grid_row, n_grid_col)
            Xc = self.milan_data[out_start_idx - self.close_len: out_start_idx]
            # 将 Xc reshape 为 (n_grid, close_len) 并转置，使每个网格的历史数据成为一列
            Xc = Xc.reshape((Xc.shape[0], Xc.shape[1] * Xc.shape[2])).transpose(1, 0)
            Xc = torch.from_numpy(Xc)
            N, C = Xc.shape
            # 通过相关性矩阵选择 top-K 的网格
            grid_map = self._grid_selection(Xc, self.K_grids)
            # 扩展维度并根据 grid_map 选择 top-K 网格数据
            Xc_expand = Xc.unsqueeze(0).expand(N, N, C)
            grids_topk = Xc_expand.gather(1, grid_map.unsqueeze(2).expand((*grid_map.shape, C)))
            self.grid_topk = grids_topk
            self.curr_slice = n_slice

        # 从缓存的 top-K 网格中提取当前网格对应的数据
        Xs = grids_topk[self.milan_data.shape[1] * n_row + n_col]

        # 提取单个网格对应的近期数据，用作输入
        idx_grid_data = self.milan_data[:, n_row, n_col]
        Xc = idx_grid_data[out_start_idx - self.close_len: out_start_idx]
        # 获取周期性历史数据的索引
        indices = get_indexes_of_train('sttran', self.time_level, out_start_idx, self.close_len, self.period_len)
        # 对于每个周期性索引，提取数据；如果索引无效则用 0 填充
        Xp = [idx_grid_data[i] if i >= 0 else 0 for i in indices]
        Xp = np.stack(Xp, axis=0).astype(np.float32)
        Xp = Xp.reshape((self.period_len, self.close_len))
        # Y 为预测目标，取出预测时刻对应的数据
        Y = idx_grid_data[out_start_idx: out_start_idx + self.pred_len]
        return Xc, Xp, Xs, Y  # 分别返回近期数据、周期数据、选定的 top-K 网格数据以及预测目标

    def _grid_selection(self, Xc, K):
        """
        根据相关性矩阵选择每个网格与其它网格的相关性，返回每个网格的 top-K 索引。
        这里采用 Pearson 相关系数计算相关性，若出现 NaN 则设为 -1，然后返回 top-K 索引。
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ncov = np.corrcoef(Xc)
            ncov[np.isnan(ncov)] = -1
        return torch.topk(torch.from_numpy(ncov), k=K, dim=1).indices


if __name__ == '__main__':

    aggr_time = '10min'
    time_range = 'all'
    tele_column = 'sms'
    milan_dataset = MilanSW(aggr_time=aggr_time,
                            tele_column=tele_column,
                            close_len=16,
                            pred_len=8,
                            format = 'normal')
    milan_dataset.prepare_data()
    milan_dataset.setup()
    train_dataloader = milan_dataset.train_dataloader()

    print("Number of batches in train_dataloader:", len(train_dataloader))

    batch = next(iter(train_dataloader))
    # 假设 batch 返回的是 (X, Y)
    X, Y = batch
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # 	•	MilanSlidingWindowDataset (normal 格式)：
    # 	•	X: (batch_size, input_len, window_size × window_size)
    # 	•	Y: (batch_size, pred_len)

    # 	•	MilanSW3CompDataset (3分量格式)：
    # 	•	Xc: (batch_size, close_len)
    # 	•	Xp: (batch_size, period_len, close_len)
    # 	•	Xs: (batch_size, K_grids, close_len)
    # 	•	Y:  (batch_size, pred_len)

    # 	•	MilanSWInformerDataset：
    # 	•	X: (batch_size, input_len, window_size × window_size)
    # 	•	Y: (batch_size, pred_len, 1)