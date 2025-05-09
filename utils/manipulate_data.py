import pandas as pd
from utils.milano_grid import _map_back
import numpy as np
from tqdm import tqdm  # 可选，添加进度条
from scipy.ndimage import convolve

def df2cell_time_array(data):
    # reshape dataframe to ndarray of size (n_timesteps, n_cells)
    data = data.reset_index()
    data = data.pivot(index='time', columns='cellid', values='internet')
    data = data.fillna(0)
    data = data.values
    # print("reshaped data to shape {}".format(data.shape))
    return data


def df2timeindex(data):
    # extract timesteps from dataframe
    cellid = data['cellid'][0]
    timesteps = data.loc[data['cellid'] == cellid]['time']
    return timesteps


def _get_grids_by_cellids(data: pd.DataFrame, cellids: list) -> pd.DataFrame:
    """
    Get the grids of the given cellids
    :param data: the dataframe
    :param cellids: the cellids
    :return: the data of cell ids
    """
    return data.loc[data['cellid'].isin(cellids)]


def _gen_cellids_by_colrow(col1: int, col2: int, row1: int, row2: int) -> list:
    """
    Generate cellid list by col and row
    :param col1: the start column
    :param col2: the end column
    :param row1: the start row
    :param row2: the end row
    :return: a list of cellids
    """
    cellids = []
    for col in range(col1, col2 + 1):
        for row in range(row1, row2 + 1):
            cellids.append(_map_back(row-1, col-1))
    return cellids


def filter_grids_data_by_colrow(data: pd.DataFrame, col1: int, col2: int, row1: int, row2: int) -> pd.DataFrame:
    """
    Get data of grids by the given col and row
    :param data: the dataframe
    :param col1: the start column
    :param col2: the end column
    :param row1: the start row
    :param row2: the end row
    :return: the data of cell ids
    """
    cellids = _gen_cellids_by_colrow(col1, col2, row1, row2)
    data = _get_grids_by_cellids(data, cellids)
    data = data.sort_values(['cellid', 'time']).reset_index(drop=True)
    return data


def fill_grid_data_nan(grid_data):
    """
    对整个 4D grid 数据 (T, C, H, W) 进行 NaN 填补。
    """
    filled = grid_data.copy()

    T, C, H, W = grid_data.shape

    for t in tqdm(range(T), desc="Filling NaNs"):
        for c in range(C):
            slice_2d = grid_data[t, c]
            if np.isnan(slice_2d).any():
                filled[t, c] = fast_fill_nan_with_local_mean(slice_2d)
    return filled


def fast_fill_nan_with_local_mean(data_2d):
    """
    更快的二维滑动窗口 NaN 填补（使用卷积），仅对单张图像。
    """
    nan_mask = np.isnan(data_2d).astype(float)
    valid_mask = ~np.isnan(data_2d)
    data_2d = np.nan_to_num(data_2d, nan=0.0)

    kernel = np.ones((3, 3), dtype=np.float32)

    # 平滑有效值之和 & 有效值个数
    value_sum = convolve(data_2d, kernel, mode='mirror')
    valid_count = convolve(valid_mask.astype(np.float32), kernel, mode='mirror')

    # 避免除以0
    local_mean = np.divide(value_sum, valid_count, out=np.zeros_like(value_sum), where=valid_count > 0)

    # 用 local_mean 填 NaN
    filled = data_2d.copy()
    filled[nan_mask == 1] = local_mean[nan_mask == 1]

    return filled
