# taiwan 数据集
import os
import h5py
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler
from datasets.BaseDataset import SpaTemData
from utils.registry import register

@register('Taiwan')
class TaiwanDataset(SpaTemData):
    def __init__(self,
                 data_dir: str = '/data/scratch/jiayin/taiwan',
                 grid_range: tuple = (0, 2, 0, 6),  # 3 x 7
                 aggr_time: str = '5min', # original: 5min
                 batch_size: int = 32,
                 normalize: bool = True,
                 user_type: str = 'pedestrian', # vehicular, stationary
                 *args, **kwargs):
        super().__init__(
            data_dir=data_dir,
            grid_range=grid_range,
            aggr_time=aggr_time,
            batch_size=batch_size,
            normalize=normalize,
            *args, **kwargs
        )

        self.user_type = user_type
        self.file_name = f"taiwan_{self.aggr_time}_T_N_3.h5"

        self.grid_range = grid_range  # (min_row, max_row, min_col, max_col)
        if grid_range is None:
            self.grid_range = (0, 2, 0, 6)
            self.n_rows = 3
            self.n_cols = 7
        else:
            self.n_rows = grid_range[1] - grid_range[0] + 1
            self.n_cols = grid_range[3] - grid_range[2] + 1
        self.N_all = self.n_rows * self.n_cols

    def _get_holidays(self) -> List[str]:
        # 2022年8月28日（星期日）至2022年9月28日（星期三），台湾的节假日
        # 画图检查过，数值上没有明显变大
        return [
            "2022-09-09", # 周五
            "2022-09-10", # 中秋节，周六
            "2022-09-11", # 周日
        ]

    def _process_raw_data(self):
        """把原始CSV合成h5文件"""
        # 读取三种类型数据
        pedestrian = pd.read_csv(os.path.join(self.data_dir, 'Pedestrian_All.csv'))
        stationary = pd.read_csv(os.path.join(self.data_dir, 'Stationary_All.csv'))
        vehicular = pd.read_csv(os.path.join(self.data_dir, 'Vehicular_All.csv'))

        # 提取时间列（5min 间隔）
        timestamps = pd.to_datetime(pedestrian['Datetime'], format='%Y/%m/%d %H:%M')

        # 删除 'Datetime' 列，剩下纯数据
        pedestrian = pedestrian.drop(columns=['Datetime']).values  # [T, N]
        stationary = stationary.drop(columns=['Datetime']).values
        vehicular = vehicular.drop(columns=['Datetime']).values

        # 确保 shape 对齐
        assert pedestrian.shape == stationary.shape == vehicular.shape

        # 聚合前的原始数据 shape: [T, N, 3]
        data = np.stack([pedestrian, stationary, vehicular], axis=-1)  # [T, N, 3]

        # 按 aggr_time 聚合数据
        if self.aggr_time not in ['5min', None]:  # 如果不是原始粒度，则需要聚合
            df = pd.DataFrame({
                'timestamp': timestamps,
                'pedestrian': list(pedestrian),
                'stationary': list(stationary),
                'vehicular': list(vehicular)
            })

            # 将每列都展开成多列（每个节点一个列）
            def expand_array_column(df, col_name):
                array_data = np.stack(df[col_name].values)
                cols = [f'{col_name}_{i}' for i in range(array_data.shape[1])]
                return pd.DataFrame(array_data, columns=cols, index=df.index)

            df_full = pd.DataFrame({'timestamp': timestamps})
            for name in ['pedestrian', 'stationary', 'vehicular']:
                df_expanded = expand_array_column(df.assign(index=df.index), name)
                df_expanded['timestamp'] = timestamps
                df_full = df_full.merge(df_expanded, on='timestamp')

            # 设为索引并重采样聚合
            df_full.set_index('timestamp', inplace=True)
            df_resampled = df_full.resample(self.aggr_time).mean()
            df_resampled.dropna(inplace=True)  # optional

            # 还原为 [T, N, 3]
            all_data = []
            for name in ['pedestrian', 'stationary', 'vehicular']:
                cols = [col for col in df_resampled.columns if col.startswith(name)]
                arr = df_resampled[cols].values  # [T, N]
                all_data.append(arr)
            data = np.stack(all_data, axis=-1)  # [T, N, 3]
            timestamps = df_resampled.index.to_series()

        # 读取坐标文件
        coords = pd.read_csv(os.path.join(self.data_dir, 'coordinates.csv'))

        # 读取距离文件
        distances = pd.read_csv(os.path.join(self.data_dir, 'distances.txt'), sep=r'\s+')

        # 保存到h5
        h5_path = os.path.join(self.data_dir, self.file_name)
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('time', data=np.array(timestamps.dt.strftime('%Y-%m-%d %H:%M:%S'), dtype='S'))
            f.create_dataset('coordinates', data=coords.values)
            f.create_dataset('distances', data=distances.values)

    def _load_data(self):
        """核心部分：加载Taiwan的h5文件，提取需要的流量特征"""

        file_path = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"H5 file not found: {file_path}, please run prepare_data first!")

        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]  # (T, N, 3)
            timestamps = f['time'][:]  # (T,)
            coordinates = f['coordinates'][:]  # (N, 3)
            distances = f['distances'][:]  # (E, 3)

        # 保存时间戳, coordinate 和 distance 信息
        self.timestamps = pd.to_datetime([ts.decode('utf-8') for ts in timestamps])
        self.coordinates = coordinates
        self.distances = distances

        # 根据用户类型选择对应数据
        if not hasattr(self, 'user_type'):
            raise ValueError("user_type must be specified in the class (e.g., 'pedestrian', 'all')")

        valid_types = ['pedestrian', 'stationary', 'vehicular', 'all', 'v-s', 'v-p']
        if self.user_type not in valid_types:
            raise ValueError(f"Invalid user_type: {self.user_type}")

        if self.user_type == 'all':
            data = data.sum(axis=-1, keepdims=True)  # shape: [T, N, 1]
        elif self.user_type == 'v-s':
            data = (data[..., 2] - data[..., 1])[..., np.newaxis]  # vehicular - stationary → shape [T, N, 1]
        elif self.user_type == 'v-p':
            data = (data[..., 2] - data[..., 0])[..., np.newaxis]  # vehicular - pedestrian
        else:
            index = {'pedestrian': 0, 'stationary': 1, 'vehicular': 2}[self.user_type]
            data = data[..., index:index + 1]  # 保留通道维度，shape: [T, N, 1]

        # 预处理grid data
        T, N, C = data.shape
        self.service_dim = C

        # Reshape成 [T, C, H, W]
        # 默认是3x7 = 21节点，需要确定排列
        self.grid_data = data.reshape(T, 3, 7, -1).transpose(0, 3, 1, 2)  # (T, C, H, W)

        self.T = self.grid_data.shape[0]

        if self.normalize:
            shape = self.grid_data.shape
            self.scaler = MinMaxScaler()
            self.grid_data = self.scaler.fit_transform(self.grid_data.reshape(-1,1)).reshape(shape)

    def _get_spatial_feature(self):
        self.spatial_feature = self.coordinates.transpose(1, 0).reshape(-1, 3, 7)
        # (3, 3, 7)


if __name__ == '__main__':
    taiwan = TaiwanDataset(format = 'default',
                           data_dir = '/data/scratch/jiayin/taiwan',
                           grid_range = (0, 2, 0, 6),
                           aggr_time = '10min',
                           batch_size = 32,
                           close_len=6,
                           pred_len= 3,
                           normalize = True,
                           user_type = 'vehicular',)
    taiwan.prepare_data()
    taiwan.setup()

    train_dl = taiwan.train_dataloader()
    val_dl = taiwan.val_dataloader()
    test_dl = taiwan.test_dataloader()

    print("Number of batches in train_dataloader:", len(train_dl))
    print("Number of batches in val_dataloader:", len(val_dl))
    print("Number of batches in test_dataloader:", len(test_dl))

    # 获取一个 batch，查看 X 和 Y 的维度
    batch = next(iter(test_dl))

    # 这里假设 _get_dataset 返回的数据结构为 (X, Y)
    X, Y = batch
    # print("Batch Xc shape:", Xc.shape)
    # print("Batch Xp shape:", Xp.shape)
    # # print("Batch Xt shape:", Xt.shape)
    print("Batch X shape:", X.shape)
    print("Batch Y shape:", Y.shape)
    # print("Batch X_timefeature shape:", X_time.shape)
    # print("Batch X_meta shape:", X_meta.shape)

