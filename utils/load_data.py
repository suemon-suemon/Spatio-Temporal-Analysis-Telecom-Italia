import pandas as pd

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