import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.manipulate_data import df2cell_time_array, filter_grids_data_by_colrow


def load_telecom_data(path):
    print("loading data from file: {}".format(path))
    data = pd.read_csv(path, header=0, index_col=0)
    data = data.groupby(['cellid', 'time'], as_index=False).sum()
    data.drop(['countrycode'], axis=1, inplace=True)
    return data


def load_multi_telecom_data(paths: list):
    """
    load and combine telecom data from multiple files, sorted by cellid and time
    Args:
        paths (list): list of paths to the files

    Returns:
        pd.Dataframe: combined data
    """
    data = pd.DataFrame()
    for path in paths:
        data = pd.concat([data, load_telecom_data(path)], ignore_index=True)
    data = data.sort_values(['cellid', 'time']).reset_index(drop=True)
    print("loaded {} rows".format(len(data)))
    return data


def load_grid_data(data_dir: str, normalize: bool = False):
    return load_part_grid_data(data_dir, normalize=normalize, col1=1, col2=100, row1=1, row2=100)


def load_part_grid_data(data_dir: str,
                        file_name: str = 'milan_telecom_data.csv.gz',
                        normalize: bool = False, 
                        aggr_time: str = None,
                        col1: int = 1, col2: int = 100, 
                        row1: int = 1, row2: int = 100):
    if aggr_time not in [None, 'hour']:
        raise ValueError("aggre_time must be None or 'hour'")
    filePath = os.path.join(data_dir, file_name)
    if not os.path.exists(filePath):
        raise FileNotFoundError("file {} not found".format(filePath))

    milan_data = pd.read_csv(filePath, compression='gzip', usecols=['cellid', 'time', 'internet'])
    milan_data['time'] = pd.to_datetime(milan_data['time'], format='%Y-%m-%d %H:%M:%S')
    milan_data = filter_grids_data_by_colrow(milan_data, col1, col2, row1, row2)
    if aggr_time == 'hour':
        milan_data = milan_data.groupby(['cellid', pd.Grouper(key="time", freq="1H")]).sum()
        milan_data.reset_index(inplace=True)
    # reshape dataframe to ndarray of size (n_timesteps, n_cells)
    milan_grid_data = df2cell_time_array(milan_data)
    
    scaler = None
    if normalize:
        scaler = MinMaxScaler((0, 10))
        milan_grid_data = scaler.fit_transform(milan_grid_data.reshape(-1, 1)).reshape(milan_grid_data.shape)
        # milan_grid_data = scaler.fit_transform(milan_grid_data)
    # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
    milan_grid_data = milan_grid_data.astype(np.float32)
    print("loaded {} rows and {} grids".format(milan_grid_data.shape[0], milan_grid_data.shape[1]))
    return milan_grid_data, milan_data, scaler # ndarray shape of (n_timesteps, n_grids), original dataframe


def load_and_save_telecom_data_by_tele(paths: list, save_path: str, tele_column: str = 'internet'):
    """
    load and combine telecom data from multiple files, sorted by cellid and time
    Args:
        paths (list): list of paths to the files
    """
    # raise(NotImplementedError)
    data = load_multi_telecom_data(paths)
    data = data[['cellid', 'time', tele_column]]
    data.to_csv(os.path.join(
        save_path, 'milan_internet_all_data.csv.gz'), compression='gzip', index=False)
    return
