from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import time

import numpy as np
import pandas as pd
import pmdarima as pm
from datasets.milan import Milan
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
from utils.load_data import load_part_grid_data
from utils.nrmse import nrmse
from utils.tqdm_joblib import tqdm_joblib


# load milan time series data
test_split_date = Milan.get_default_split_date()['test']
milan_grid_data, milan_df = load_part_grid_data('../data/sms-call-internet-mi', col1=41, col2=70, row1=41, row2=70)

train_len = milan_df['time'][milan_df['time'] < pd.Timestamp(**test_split_date)].unique().shape[0]
milan_train, milan_test = Milan.train_test_split(milan_grid_data, train_len, is_val=False)

n_series = milan_train.shape[1]
predict_step = 1


def ARIMA_step_forecast(train, test, predict_step=1):
    arima_model = pm.ARIMA(order=(12, 1, 2), suppress_warnings=True)
    arima_model.fit(train)
    forecasts = []
    def forecast_one_step():
        fc, _ = arima_model.predict(n_periods=1, return_conf_int=True)
        return fc.tolist()[0]

    for new_ob in test:
        fc = forecast_one_step()
        forecasts.append(fc)
        # Updates the existing model with a small number of MLE steps
        arima_model.update(new_ob)
    return forecasts

tic = time.time()
with tqdm_joblib(tqdm(desc="ARIMA predict", total=n_series)) as progress_bar:
    grid_pred = Parallel(n_jobs=128)(delayed(ARIMA_step_forecast)(milan_train[:, i], milan_test[:, i], predict_step) for i in range(n_series))
grid_pred = np.array(grid_pred).T
toc = time.time()
print('Parallel ARIMA Runing time:', toc - tic)

# tic = time.time()
# ## ******* Single Thread version ******* 
# grid_pred2 = []
# for grid in tqdm(range(n_series)):
#     train, test = milan_train[:, grid], milan_test[:, grid]
#     forecasts = ARIMA_step_forecast(train, test, predict_step)
#     grid_pred2.append(forecasts)
# grid_pred2 = np.array(grid_pred2).T
# ## ******* Single Thread version ******* 
# toc = time.time()
# print('SingleThread ARIMA Runing time:', toc - tic)

np.save('../data/milan_arima_pred_step1.npy', grid_pred)
mae = np.mean([mean_absolute_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
mape = np.mean([mean_absolute_percentage_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
nrmse = nrmse(milan_test, grid_pred)

print('MAE: ', mae, ' MAPE: ', mape, ' NRMSE: ', nrmse)