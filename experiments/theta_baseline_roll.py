from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import time

import numpy as np
import pandas as pd
from datasets.milan import Milan
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm

from utils.load_data import load_part_grid_data
from utils.nrmse import nrmse


# load milan time series data
test_split_date = Milan.get_default_split_date()['test']
milan_grid_data, milan_df = load_part_grid_data('../data/sms-call-internet-mi', col1=41, col2=70, row1=41, row2=70)

train_len = milan_df['time'][milan_df['time'] < pd.Timestamp(**test_split_date)].unique().shape[0]
milan_train, milan_test = Milan.train_test_split(milan_grid_data, train_len, is_val=False)

n_series = milan_train.shape[1]
predict_step = 1

tic = time.time()
grid_pred = []
with tqdm(total=n_series * 1440 / predict_step) as pbar:
    for grid in tqdm(range(n_series)):
        series = milan_grid_data[:, grid]
        train, test = milan_train[:, grid], milan_test[:, grid]
        forecasts = []
        for i in tqdm(range(len(test))):
            sub_model = ThetaModel(np.concatenate((train, test[:i])), period=144).fit()
            forecast = sub_model.forecast(predict_step)
            forecasts.append(forecast.values[0])
            pbar.update(1)
        
        grid_pred.append(forecasts)

toc = time.time()
grid_pred = np.array(grid_pred).T

print('Theta Runing time:', toc - tic)

np.save('../data/milan_theta_pred_step1.npy', grid_pred)
mae = np.mean([mean_absolute_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
mape = np.mean([mean_absolute_percentage_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
nrmse = nrmse(milan_test, grid_pred)

print('MAE: ', mae, ' MAPE: ', mape, ' NRMSE: ', nrmse)
