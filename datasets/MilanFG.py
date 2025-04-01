import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dtaidistance import dtw

from datasets.Milan import Milan, get_indexes_of_train
from utils.time_features import time_features
from torch.utils.data.sampler import RandomSampler

class MilanFG(Milan):
    """ Milan Dataset in a full-grid fashion """
    """
        Milan 数据集（全网格方式）的 DataModule 版本

        该类继承自 Milan DataModule，用于构造全网格数据集，即在整个空间网格上进行预测任务。
        数据通过滑动窗口方式分割为训练、验证和测试集，并同时构造相应的时间戳信息。

        参数:
          - format (str): 数据集格式，可选 'default', 'informer', 'sttran', 'stgcn', 'timeF'。
                          默认 'default'。该参数决定了后续数据加载时采用哪种 Dataset 实现。
          - close_len (int): 最近历史窗口的时间步数，用于作为输入的历史数据。比如 12 表示用 12 个时间步。
          - period_len (int): 周期性历史窗口的时间步数，用于捕捉周期性特征，默认 0 表示不考虑周期性。
          - trend_len (int): 趋势数据窗口的时间步数，默认 0 表示不考虑趋势信息。
          - label_len (int): 标签窗口长度，部分格式（如 informer）中会用到，用于生成时间特征标签。默认 0。
          - pred_len (int): 预测未来时间步的数量，默认 1。
          - **kwargs: 其它参数传递给基类 Milan，例如 data_dir, aggr_time, tele_column, time_range, compare_mvstgn, load_meta, impute_missing 等。

        行为:
          - prepare_data(): 调用基类方法进行数据下载和预处理（如构造 HDF5 文件、归一化、切分网格数据等）。
          - setup(stage): 根据预处理结果划分训练、验证、测试集，并生成相应的时间戳字典 self.milan_timestamps 。
                         同时通过 train_test_split 方法切分 self.milan_grid_data，构造出 self.milan_train、self.milan_val 和 self.milan_test。
                         注：对于测试集，还额外拼接了验证集末尾的一部分数据以确保滑动窗口连续性。
          - train_dataloader(), val_dataloader(), test_dataloader(), predict_dataloader():
                         分别返回训练、验证、测试和预测阶段的 DataLoader。

        输入输出维度:
          - 内部存储的 milan_grid_data 的形状通常为 (n_timestamps, n_grid_row, n_grid_col)。
          - 划分后的训练/验证/测试数据保留了相同的空间维度，时间维度根据滑动窗口的设置进行切分。
        """
    def __init__(self, 
                 format: str = 'default',
                 close_len: int = 12, 
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 0,
                 pred_len: int = 1,
                 **kwargs):
        super(MilanFG, self).__init__(**kwargs)
        self.format = format
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.label_len = label_len
        self.pred_len = pred_len

    def prepare_data(self):
        Milan.prepare_data(self)
    
    def setup(self, stage=None):
        self.prepare_data() # 以初始化self.meta
        Milan.setup(self, stage)

        train_len, val_len, test_len = self.get_default_len()

        self.milan_timestamps = {
            "train": self.timestamps[:train_len],
            "val": self.timestamps[train_len:train_len+val_len],
            "test": self.timestamps[train_len+val_len-(self.close_len+self.pred_len-1):train_len+val_len+test_len],
        }

        self.milan_train, self.milan_val, self.milan_test = self.train_test_split(self.milan_grid_data, train_len, val_len, test_len)

        self.mialn_val = np.concatenate((self.milan_train[-(self.close_len+self.pred_len-1):], self.milan_val))
        self.milan_test = np.concatenate((self.milan_val[-(self.close_len+self.pred_len-1):], self.milan_test))
        # print('train shape: {}, val shape: {}, test shape: {}'.format(self.milan_train.shape, self.milan_val.shape, self.milan_test.shape))

    def train_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_train, 'train', self.meta), batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_val, 'val', self.meta), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self._get_dataset(self.milan_test, 'test', self.meta), batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_dataset(self, data, stage, meta=None):
        # print('milan_data_shape: ', data.shape)
        # milan_data_shape: (6796, 1, 20, 20)

        if self.format == 'default':
            return MilanFullGridDataset(data, self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.pred_len)
        elif self.format == 'mywat':
            return MilanMyWATDataset(data, self.aggr_time, self.close_len, self.period_len,
                                     self.trend_len, self.pred_len)
        elif self.format == 'informer':
            return MilanFGInformerDataset(data, self.milan_timestamps[stage], self.aggr_time, self.close_len, 
                                            self.period_len, self.trend_len, self.label_len, self.pred_len)
        elif self.format == 'sttran':
            return MilanFGStTranDataset(data, self.aggr_time, self.close_len, self.period_len, self.pred_len)
        elif self.format == 'stgcn':
            return MilanFGStgcnDataset(data, self.aggr_time, self.close_len, self.period_len, self.trend_len, self.pred_len)
        elif self.format == 'timeF':
            return MilanFGTimeFDataset(data, meta, self.milan_timestamps[stage], self.aggr_time, self.close_len, self.period_len, self.trend_len, self.pred_len)
        elif self.format == 'scope':
            return MilanSCOPEDataset(data, self.aggr_time, self.close_len, self.period_len,
                                     self.trend_len, self.pred_len)

class MilanFullGridDataset(Dataset):
    """full grid"""
    """
    全网格滑动窗口数据集（默认格式）

    该类采用滑动窗口方法构造 Milan 数据集，适用于全网格预测任务。
    每个样本通过选择一个时间窗口内的数据来作为输入 X，并将预测时刻单个网格的值作为目标 Y。

    输入数据:
      - milan_data: 一个 pandas DataFrame 或 NumPy 数组，形状为 (n_timestamps, n_grid_row, n_grid_col)
                    表示在不同时间步、不同空间网格上的数值记录。

    参数:
      - milan_data: 上述输入数据。
      - aggr_time (str): 时间聚合方式，要求为 None 或 'hour'，保证时间分组一致性。
      - close_len (int): 最近历史数据窗口的时间步数，即用于输入的历史数据长度。
      - period_len (int): 周期性历史数据窗口的时间步数（可选），默认 0 表示不使用周期性数据。
      - trend_len (int): 趋势数据窗口的时间步数（可选），默认 0 表示不使用趋势信息。
      - pred_len (int): 预测窗口长度，即要预测未来的时间步数，通常为 1。

    返回:
      - __getitem__ 返回一个元组 (X, Y)：
          * X: 输入数据窗口，Batch X 形状为 (batch_size, close_len, services, n_grid_row, n_grid_col)。
               其中 input_len 通常等于 close_len。窗口数据来自填充后的数据，
               通过滑动窗口在空间维度（n_grid_row, n_grid_col）中截取局部区域。
          * Y: 目标值，取自预测时刻单个网格的值，Batch Y 形状为
               (batch_size, pred_len, services, n_grid_row, n_grid_col)

    计算样本数:
      - 样本数 = (n_timestamps - input_len - pred_len + 1) * n_grid_row * n_grid_col，
        即在时间维度上滑动窗口的个数乘以空间中所有网格的数量。

    额外说明:
      - 该类在初始化时使用 np.pad 对原始数据进行空间填充：对时间维度不填充，
        对空间维度在前后各填充 window_size//2 个单位，填充值为 0，以确保边缘区域也能构造出完整窗口。
    """
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len
        print("MilanFullGridDataset: length {}".format(self.__len__()))

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)

        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0).astype(np.float32)
        # X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len].astype(np.float32)
        return X, Y


class MilanFGInformerDataset(Dataset):
    """
    针对Informer模型的 Milan 全网格数据集。
    除了基本的输入和目标数据外，还会生成相应的时间特征。

    参数：
      - milan_data (pd.DataFrame): 原始 Milan 数据，形状 (n_timestamps, n_grid_row, n_grid_col)。
      - timestamps (pd.DataFrame): 与数据对应的时间戳数据，用于生成时间特征。
      - aggr_time: 时间聚合方式，例如 'hour' 或其它（决定时间特征生成频率）。
      - window_size (int): 空间窗口大小，用于截取局部区域数据。
      - input_len (int): 输入数据的时间步数。
      - label_len (int): 标签窗口长度，用于构建时间特征标签。
      - pred_len (int): 预测窗口长度，表示未来预测的时间步数。

    返回：
      - X: 输入数据，形状为 (input_len, window_size*window_size)。
      - Y: 目标数据，形状为 (pred_len, 1)。
      - X_timefeature: 输入对应的时间特征，形状由 time_features 函数决定。
      - Y_timefeature: 目标对应的时间特征，形状由 time_features 函数决定。
    """
    def __init__(self,
                 milan_data,
                 timestamps,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 label_len: int = 12,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.timestamps = time_features(timestamps, timeenc=1, 
                                        freq='h' if self.time_level == 'hour' else 't')
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len)
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0)
        X = X.reshape((X.shape[0], X.shape[2] * X.shape[3])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx-self.label_len: out_start_idx+self.pred_len].squeeze()
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

        X_timefeature = [self.timestamps[i] if i >= 0 else np.zeros((self.timestamps.shape[1])) for i in indices]
        X_timefeature = np.stack(X_timefeature, axis=0)
        Y_timefeature = self.timestamps[out_start_idx-self.label_len: out_start_idx+self.pred_len]

        return X, Y, X_timefeature, Y_timefeature
    

class MilanFGTimeFDataset(Dataset):
    """
    针对 Time Feature 模型输入构建的 Milan 全网格数据集。
    除了输入和目标数据，还返回了辅助的 meta 数据和时间特征。

    参数：
      - milan_data (pd.DataFrame): 原始 Milan 数据，形状 (n_timestamps, n_grid_row, n_grid_col)。
      - meta: 网格元数据，用于辅助信息。
      - timestamps (pd.DataFrame): 时间戳数据，用于生成时间特征。
      - aggr_time (str): 时间聚合方式，决定时间特征生成的频率。
      - close_len (int): 最近历史数据窗口长度。
      - period_len (int): 周期性历史数据窗口长度。
      - trend_len (int): 趋势数据窗口长度（如有）。
      - pred_len (int): 预测窗口长度。

    返回：
      - X: 输入数据，形状为 (batch_size, close_len, services, n_grid_row, n_grid_col)
      - Y: 目标数据，形状为 (batch_size, pred_len, services, n_grid_row, n_grid_col)
      - X_timefeature: 输入数据对应的时间特征。
      - X_meta: 网格元数据，通常与空间信息相关。
"""
    def __init__(self,
                 milan_data,
                 meta,
                 timestamps,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.meta = meta

        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len
        self.timestamps = time_features(timestamps, timeenc=1, 
                                freq='h' if self.time_level == 'hour' else 't')
        print("MilanFullGridDataset: length {}".format(self.__len__()))

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]
        indices = get_indexes_of_train('default', self.time_level, out_start_idx, 
                                        self.close_len, self.period_len, self.trend_len, pred_len=self.pred_len)

        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        X = np.stack(X, axis=0).astype(np.float32)
        # dist = dtw.distance_matrix_fast(X.reshape(self.in_len, -1).T.astype(np.float64))
        # X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2])) # (n_features, n_timestamps, n_grid_row, n_grid_col)))

        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len].astype(np.float32)

        X_timefeature = [self.timestamps[i] if i >= 0 else np.zeros((self.timestamps.shape[1])) for i in indices]
        X_timefeature = np.stack(X_timefeature, axis=0).astype(np.float32)
        X_meta = self.meta.astype(np.float32)
        return X, Y, X_timefeature, X_meta


class MilanFGStTranDataset(Dataset):
    """
        STTran 版本的 Milan 全网格数据集。
        此数据集用于 STTran 模型，通过近期数据、周期性数据来构造输入，并预测未来的值。

        参数：
          - milan_data: 原始 Milan 数据，形状 (n_timestamps, n_grid_row, n_grid_col)。
          - aggr_time (str): 时间聚合方式，支持 None 或 'hour'。
          - close_len (int): 最近历史数据窗口长度。
          - period_len (int): 周期性历史数据窗口长度。
          - pred_len (int): 预测窗口长度，默认 3。

        返回：
          - Xc: 最近历史数据，处理后形状为 (n_features, close_len)。
          - Xp: 周期性历史数据，处理后形状为 (n_features, period_len)。
          - Y: 预测目标，处理后形状为 (n_features, pred_len)。
    """
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 3,
                 period_len: int = 3,
                 pred_len: int = 3):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.in_len = close_len
        self.pred_len = pred_len


    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len
    
    def __getitem__(self, idx):
        out_start_idx = idx + self.in_len
        slice_shape = self.milan_data.shape[1:]

        Xc = self.milan_data[out_start_idx-self.close_len: out_start_idx] # Xc
        indices = get_indexes_of_train('sttran', self.time_level, out_start_idx, self.close_len, self.period_len)
        Xp = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        Xp = np.stack(Xp, axis=0).astype(np.float32)
        Y = self.milan_data[out_start_idx: out_start_idx+self.pred_len]

        Xc = Xc.reshape((Xc.shape[0], Xc.shape[1] * Xc.shape[2])).transpose(1, 0)
        Xp = Xp.reshape((self.period_len, self.close_len, Xp.shape[1] * Xp.shape[2])).transpose(2, 1, 0)
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2])).transpose(1, 0)
        return Xc, Xp, Y # (N, c), (N, p, c), (N, c)


class MilanFGStgcnDataset(Dataset):
    """
        STGCN 版本的 Milan 全网格数据集。
        此版本主要用于 STGCN 模型输入构造，通过历史数据预测未来值，不区分周期或趋势部分。

        参数：
          - milan_data: 原始 Milan 数据，形状 (n_timestamps, n_grid_row, n_grid_col)。
          - aggr_time (str): 时间聚合方式，通常为 None 或 'hour'。
          - close_len (int): 最近历史数据窗口长度。
          - period_len (int): 周期性数据窗口长度（可为0）。
          - trend_len (int): 趋势数据窗口长度（可为0）。
          - pred_len (int): 预测窗口长度。

        返回：
          - Xc: X_close [batch_size, n_grid_row * n_grid_col, 1, close_len]
          - Xp: X_period [batch_size, n_grid_row * n_grid_col, 1, period_len]
          - Xt: X_trend [batch_size, n_grid_row * n_grid_col, 1, trend_len]
          - Y: [batch_size, pred_len, 1, n_grid_row, n_grid_col]
        """
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len

    def __getitem__(self, idx):
        # 计算预测起始时间点的索引，通常 idx 表示样本编号，
        # 而 self.in_len 表示用于输入历史数据的长度，因此 out_start_idx 为历史数据之后的第一个预测时间点
        out_start_idx = idx + self.in_len

        # 获取单个时间步数据的空间形状，例如 (n_grid_row, n_grid_col)
        slice_shape = self.milan_data.shape[1:]

        # 计算需要的时间索引列表，用于获取历史数据窗口。
        # get_indexes_of_train 是一个辅助函数，根据传入的参数（时间级别、起始索引、历史长度、周期长度、趋势长度）
        # 返回一组时间索引，通常这些索引指向历史数据点
        indices = get_indexes_of_train('default', self.time_level, out_start_idx,
                                       self.close_len, self.period_len, self.trend_len)
        # 将索引列表反转（可能是为了让数据顺序从最早到最新）
        indices.reverse()

        # 利用列表推导构造输入数据 X
        # 对于列表中的每个索引，如果索引 i 小于 0，则填充一个与单个时间步数据同样形状的全零数组；
        # 否则取 self.milan_data[i] 对应的时间步数据
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        # 将列表 X 堆叠成一个 NumPy 数组，并转换为 float32 类型
        X = np.stack(X, axis=0).astype(np.float32)
        # 将 X reshape 成三维数组，其形状变为 (n_timestamps, 1, n_grid_row * n_grid_col)
        # 这里将每个时间步的二维空间数据展平成一维，第二维保留为单个通道
        X = X.reshape((X.shape[0], -1, X.shape[3] * X.shape[2]))

        # 获取预测目标 Y，从 out_start_idx 开始，取 self.pred_len 个连续时间步的数据
        Y = self.milan_data[out_start_idx: out_start_idx + self.pred_len]

        # 处理 close（最近历史数据）部分：
        # X[:self.close_len] 取输入数据中前 close_len 个时间步，
        # 然后转置，使其形状变为 (n_grid_row * n_grid_col, 1, close_len)
        Xc = X[:self.close_len].transpose(2, 1, 0)

        # 根据 period_len 和 trend_len 的设置返回不同的输入组合：
        # 如果周期和趋势长度均为 0，则仅返回 close 部分 Xc 和目标 Y
        if self.period_len == 0 and self.trend_len == 0:
            return [Xc], Y
        # 如果趋势长度为 0，仅返回 close 部分和周期部分
        elif self.trend_len == 0:
            # 取出周期部分的数据，即从 close_len 开始，取 period_len 个时间步，
            # 然后转置，使其形状变为 (n_grid_row * n_grid_col, 1, period_len)
            Xp = X[self.close_len: self.close_len + self.period_len].transpose(2, 1, 0)
            return [Xc, Xp], Y
        else:
            # 如果同时存在周期和趋势部分，
            # 先提取周期部分：从 close_len 开始，取 period_len 个时间步
            Xp = X[self.close_len: self.close_len + self.period_len].transpose(2, 1, 0)
            # 再提取趋势部分：紧跟周期部分之后，取 trend_len 个时间步
            Xt = X[self.close_len + self.period_len: self.close_len + self.period_len + self.trend_len].transpose(2, 1, 0)
            return [Xc, Xp, Xt], Y


class MilanMyWATDataset(Dataset):
    # X: [batch_size, n_grid*n_row, close_len]
    # Y: [batch_size, n_grid*n_row, pred_len]
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 12,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 1):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len

    def __getitem__(self, idx):

        # 计算预测起始时间点的索引，通常 idx 表示样本编号，
        # 而 self.in_len 表示用于输入历史数据的长度，因此 out_start_idx 为历史数据之后的第一个预测时间点
        out_start_idx = idx + self.in_len

        # 获取单个时间步数据的空间形状，例如 (n_grid_row, n_grid_col)
        slice_shape = self.milan_data.shape[1:]

        # 计算需要的时间索引列表，用于获取历史数据窗口。
        # get_indexes_of_train 是一个辅助函数，根据传入的参数（时间级别、起始索引、历史长度、周期长度、趋势长度）
        # 返回一组时间索引，通常这些索引指向历史数据点
        indices = get_indexes_of_train('default', self.time_level, out_start_idx,
                                       self.close_len, self.period_len, self.trend_len)
        # 将索引列表反转（可能是为了让数据顺序从最早到最新）
        indices.reverse()

        # 利用列表推导构造输入数据 X
        # 对于列表中的每个索引，如果索引 i 小于 0，则填充一个与单个时间步数据同样形状的全零数组；
        # 否则取 self.milan_data[i] 对应的时间步数据
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        # 将列表 X 堆叠成一个 NumPy 数组，并转换为 float32 类型
        X = np.stack(X, axis=0).astype(np.float32)
        # 将 X reshape 成三维数组，其形状变为 (n_timestamps, 1, n_grid_row * n_grid_col)
        # 这里将每个时间步的二维空间数据展平成一维，第二维保留为单个通道
        X = X.reshape((X.shape[0], 1, X.shape[3] * X.shape[2])).squeeze().transpose(1, 0)

        # 获取预测目标 Y，从 out_start_idx 开始，取 self.pred_len 个连续时间步的数据
        Y = self.milan_data[out_start_idx: out_start_idx + self.pred_len]
        Y = Y.reshape((Y.shape[0], 1, Y.shape[3] * Y.shape[2])).squeeze().transpose(1, 0)
        return X,Y

class MilanSCOPEDataset(Dataset):
    # X: [batch_size, n_grid*n_row, close_len, services]
    # Y: [batch_size, n_grid*n_row, pred_len, services]
    def __init__(self,
                 milan_data,
                 aggr_time: str,
                 close_len: int = 16,
                 period_len: int = 0,
                 trend_len: int = 0,
                 pred_len: int = 8):
        # 3d array of shape (n_timestamps, n_grid_row, n_grid_col)
        # if aggr_time not in [None, 'hour']:
        #     raise ValueError("aggre_time must be None or 'hour'")
        self.time_level = aggr_time
        self.milan_data = milan_data
        self.close_len = close_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.in_len = close_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.milan_data)+1 - self.in_len - self.pred_len

    def __getitem__(self, idx):

        # 计算预测起始时间点的索引，通常 idx 表示样本编号，
        # 而 self.in_len 表示用于输入历史数据的长度，因此 out_start_idx 为历史数据之后的第一个预测时间点
        out_start_idx = idx + self.in_len

        # 获取单个时间步数据的空间形状，例如 (n_grid_row, n_grid_col)
        slice_shape = self.milan_data.shape[1:]

        # 计算需要的时间索引列表，[close_len, period_len, trend_len]。
        # get_indexes_of_train 是一个辅助函数，根据传入的参数（时间级别、起始索引、历史长度、周期长度、趋势长度）
        # 返回一组时间索引，通常这些索引指向历史数据点
        indices = get_indexes_of_train('default', self.time_level, out_start_idx,
                                       self.close_len, self.period_len, self.trend_len)
        # 将索引列表反转（可能是为了让数据顺序从最早到最新）
        indices.reverse()

        # 利用列表推导构造输入数据 X
        # 对于列表中的每个索引，如果索引 i 小于 0，则填充一个与单个时间步数据同样形状的全零数组；
        # 否则取 self.milan_data[i] 对应的时间步数据
        X = [self.milan_data[i] if i >= 0 else np.zeros(slice_shape) for i in indices]
        # 将列表 X 堆叠成一个 NumPy 数组，并转换为 float32 类型
        X = np.stack(X, axis=0).astype(np.float32)
        # 将 X reshape 成三维数组，其形状变为 (n_timestamps, 1, n_grid_row * n_grid_col)
        X = X.reshape((X.shape[0], -1, X.shape[3] * X.shape[2])).transpose(2, 0, 1)

        # 获取预测目标 Y，从 out_start_idx 开始，取 self.pred_len 个连续时间步的数据
        Y = self.milan_data[out_start_idx: out_start_idx + self.pred_len]
        Y = Y.reshape((Y.shape[0], -1, Y.shape[3] * Y.shape[2])).transpose(2, 0, 1)
        return X,Y

if __name__ == '__main__':
    # 此处假设你已经构造了一个 MilanFG 数据集实例，传入必要参数
    # 参数中的 normalize, aggr_time, time_range, tele_column 等传给基类 Milan
    milan_dataset = MilanFG(#format='',
                            batch_size=32,
                            aggr_time=None,
                            time_range='all',
                            tele_column='sms',
                            close_len = 32, # => Xc (stgcn)
                            period_len = 96, # => Xp (stgcn)
                            trend_len = 0, # => Xt (stgcn)
                            pred_len = 32,
                            )
    milan_dataset.prepare_data()
    milan_dataset.setup()

    # 获取 train_dataloader
    train_dl = milan_dataset.train_dataloader()
    val_dl = milan_dataset.val_dataloader()
    test_dl = milan_dataset.test_dataloader()

    print("Number of batches in train_dataloader:", len(train_dl))
    print("Number of batches in val_dataloader:", len(val_dl))
    print("Number of batches in test_dataloader:", len(test_dl))

    # 获取一个 batch，查看 X 和 Y 的维度
    batch = next(iter(train_dl))

    # 这里假设 _get_dataset 返回的数据结构为 (X, Y)
    X, Y = batch
    print(len(X))
    # print("Batch Xc shape:", Xc.shape)
    # print("Batch Xp shape:", Xp.shape)
    # # print("Batch Xt shape:", Xt.shape)
    print("Batch X shape:", X.shape)
    # print("Batch Y shape:", Y.shape)

    A = milan_dataset.adj_mx
