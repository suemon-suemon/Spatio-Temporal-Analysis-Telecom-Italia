import pandas as pd
from utils.milano_grid import map_back


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
            cellids.append(map_back(row-1, col-1))
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
