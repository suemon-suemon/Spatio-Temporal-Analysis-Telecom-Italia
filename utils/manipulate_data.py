# reshape dataframe to ndarray of size (n_cells, n_timesteps)
def df2cell_time_array(data):
    data = data.reset_index()
    data = data.pivot(index='time', columns='cellid', values='internet')
    data = data.values
    return data

# extract timesteps from dataframe
def df2timeindex(data):
    timesteps = data.loc[data['cellid'] == 1]['time']
    return timesteps