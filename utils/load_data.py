import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

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


def load_part_grid_data(data_dir: str, normalize: bool = False, col1: int = 1, col2: int = 100, row1: int = 1, row2: int = 100):
    if not os.path.exists(os.path.join(data_dir, 'milan_telecom_data.csv.gz')):
        raise FileNotFoundError("file {} not found".format(data_dir))
    milan_data = pd.read_csv(os.path.join(
        data_dir, 'milan_telecom_data.csv.gz'), compression='gzip')
    milan_data['time'] = pd.to_datetime(
        milan_data['time'], format='%Y-%m-%d %H:%M:%S')
    milan_data = filter_grids_data_by_colrow(milan_data, col1, col2, row1, row2)
    milan_grid_data = df2cell_time_array(milan_data)
    if normalize:
        scaler = MaxAbsScaler()
        milan_grid_data = scaler.fit_transform(milan_grid_data)
    # Input and parameter tensors are not the same dtype, found input tensor with Double and parameter tensor with Float
    milan_grid_data = milan_grid_data.astype(np.float32)
    print("loaded {} rows and {} grids".format(milan_grid_data.shape[0], milan_grid_data.shape[1]))
    return milan_grid_data, milan_data # (n_timesteps, n_grids) and original dataframe


def load_and_save_telecom_data_by_tele(paths: list, save_path: str, tele_column: str = 'internet'):
    """
    load and combine telecom data from multiple files, sorted by cellid and time
    Args:
        paths (list): list of paths to the files
    """
    data = load_multi_telecom_data(paths)
    data = data[['cellid', 'time', tele_column]]
    data.to_csv(os.path.join(
        save_path, 'milan_telecom_data.csv.gz'), compression='gzip')
    return
