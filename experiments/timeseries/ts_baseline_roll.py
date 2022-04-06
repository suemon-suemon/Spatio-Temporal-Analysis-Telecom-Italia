from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import argparse
import time

import numpy as np
import pandas as pd
import pmdarima as pm
from datasets.milan import Milan
from joblib import Parallel, delayed
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm
from utils.load_data import load_part_grid_data
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

    # load milan time series data
    test_split_date = Milan.get_default_split_date()['test']
    milan_grid_data, milan_df = load_part_grid_data('../data/sms-call-internet-mi', col1=41, col2=70, row1=41, row2=70)

    train_len = milan_df['time'][milan_df['time'] < pd.Timestamp(**test_split_date)].unique().shape[0]
    milan_train, milan_test = Milan.train_test_split(milan_grid_data, train_len, is_val=False)

    n_series = milan_train.shape[1]
    predict_step = 1

    if args.method == 'theta':
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
                series = milan_grid_data[:, grid]
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

    np.save('../data/milan_{}_pred_step1.npy'.format(args.method), grid_pred)
    mae = np.mean([mean_absolute_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
    mape = np.mean([mean_absolute_percentage_error(milan_test[i, :], grid_pred[i, :]) for i in range(n_series)])
    nrmse = nrmse(milan_test, grid_pred)

    print('Method: ', args.method, 'MAE: ', mae, ' MAPE: ', mape, ' NRMSE: ', nrmse)
