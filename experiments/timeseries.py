from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import argparse
import torch
import time

import numpy as np
import pandas as pd
import pmdarima as pm
from datasets import Milan
from joblib import Parallel, delayed
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.forecasting.theta import ThetaModel
# from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
from utils.nrmse import nrmse
from utils.tqdm_joblib import tqdm_joblib


def stan_init(m):
    """Retrieve parameters from a trained Prophet model.
    
    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.
    
    Parameters
    ----------
    m: A trained model of the Prophet class.
    
    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rolling forward timeseries baseline runner')
    parser.add_argument('--method', help='baseline method')
    args = parser.parse_args()
    time_range = 'all'
    aggr_time = 'hour'

    ## prepare data
    milan = Milan(time_range=time_range, aggr_time=aggr_time, tele_column='internet')
    milan.prepare_data()
    milan.setup()
    milan_train = np.concatenate((milan.milan_train, milan.milan_val), axis=0)
    milan_test = milan.milan_test
    milan_train = milan_train.reshape((milan_train.shape[0], -1))
    milan_test = milan_test.reshape((milan_test.shape[0], -1))
    milan_data = np.concatenate((milan_train, milan_test), axis=0)

    print("Shape of train: {}, test: {}.".format(milan_train.shape, milan_test.shape))
    n_series = milan_train.shape[1]
    predict_step = 1

    if args.method == 'naive':
        grid_pred = np.concatenate((milan_train[-1:, :], milan_test[:-1, :]), axis=0)

    if args.method == 'theta':
        tic = time.time()
        grid_pred = []
        with tqdm(total=n_series * 1440 / predict_step) as pbar:
            for grid in tqdm(range(n_series)):
                series = milan_data[:, grid]
                train, test = milan_train[:, grid], milan_test[:, grid]
                forecasts = []
                for i in tqdm(range(len(test))):
                    sub_model = ThetaModel(np.concatenate((train, test[:i])), period=144).fit()
                    forecast = sub_model.forecast(predict_step)
                    forecasts.append(forecast.values[0])
                    pbar.update(1)
                grid_pred.append(forecasts)
        grid_pred = np.array(grid_pred).T
        toc = time.time()
        print('Theta Runing time:', toc - tic)

    if args.method == 'arima':
        tic = time.time()
        with tqdm_joblib(tqdm(desc="ARIMA predict", total=n_series)) as progress_bar:
            grid_pred = Parallel(n_jobs=128)(delayed(ARIMA_step_forecast)(milan_train[:, i], milan_test[:, i], predict_step) for i in range(n_series))
        grid_pred = np.array(grid_pred).T
        toc = time.time()
        print('Parallel ARIMA Runing time:', toc - tic)

    if args.method == 'prophet':
        tic = time.time()
        grid_pred = []
        with tqdm(total=n_series * 1440 / predict_step) as pbar:
            for grid in tqdm(range(n_series)):
                series = milan_data[:, grid]
                train, test = milan_train[:, grid], milan_test[:, grid]
                forecasts = []
                base_model = Prophet().fit(train)
                for i in tqdm(range(len(test))):
                    sub_model = Prophet().fit(np.concatenate((train, test[:i])), init=stan_init(base_model))
                    forecast = sub_model.predict(test[i:i+predict_step])
                    forecasts.append(forecast['yhat'].values[0])
                    pbar.update(1)
                grid_pred.append(forecasts)
        grid_pred = np.array(grid_pred).T
        toc = time.time()
        print('Prophet Runing time:', toc - tic)

    np.save('experiments/results/milan_{}_{}_{}_{}_pred_step1.npy'.format(args.method, 'internet', time_range, aggr_time), grid_pred)
    mae = np.mean([mean_absolute_error(milan_test[:, i], grid_pred[:, i]) for i in range(n_series)])
    mape = np.mean([mean_absolute_percentage_error(milan_test[:, i], grid_pred[:, i]) for i in range(n_series)])
    rmse = mean_squared_error(milan_test, grid_pred, squared=False)
    nrmse = nrmse(milan_test, grid_pred)

    print('Method: ', args.method, 'MAE: ', mae, ' MAPE: ', mape, 'RMSE: ', rmse,' NRMSE: ', nrmse)
