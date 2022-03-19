import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.metrics import mase, rmse
from darts.models import ARIMA, NaiveDrift, NaiveSeasonal, NBEATSModel, ExponentialSmoothing, AutoARIMA, FourTheta
from darts.utils.utils import SeasonalityMode
from sklearn.preprocessing import MaxAbsScaler
from numpy.random import default_rng
from tqdm import tqdm

from utils.load_data import load_multi_telecom_data
from utils.manipulate_data import df2cell_time_array, df2timeindex

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
TRAIN_VAL_SPLIT = {'year': 2013, 'month': 11, 'day': 21, 'hour': 0}
PREDICT_LEN = 6 * 24 * 10

# get files from 11-01 to 11-30
paths = ['/home/zzw/dataset/TeleMilan/sms-call-internet-mi/sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) for i in range(1, 10)] +\
        ['/home/zzw/dataset/TeleMilan/sms-call-internet-mi/sms-call-internet-mi-2013-11-{i}.csv'.format(i=i) for i in range(10, 31)]
data = load_multi_telecom_data(paths)

internet_data = df2cell_time_array(data)
timeindex = pd.DatetimeIndex(df2timeindex(data))
internet_series = TimeSeries.from_times_and_values(times=timeindex, values=internet_data)

scaler = Scaler(MaxAbsScaler())
filler = MissingValuesFiller()
internet_series = scaler.fit_transform(internet_series)
internet_series = filler.transform(internet_series)
train, val = internet_series.split_before(pd.Timestamp(
    year=TRAIN_VAL_SPLIT['year'], 
    month=TRAIN_VAL_SPLIT['month'], 
    day=TRAIN_VAL_SPLIT['day'], 
    hour=TRAIN_VAL_SPLIT['hour'], minute=0, second=0)
)
print("Timesteps of train: {}, val: {}. Total components: {}".format(len(train), len(val), train.n_components))

rng = default_rng()
grids = rng.choice(10000, size=1000, replace=False)

# tic = time.time()
# es_mases = []
# es_rmses = []
# for c in tqdm(grids):
#     c = int(c)
#     series = internet_series.univariate_component(c)
#     train_s = train.univariate_component(c)
#     model = AutoARIMA()
#     model.fit(train_s)
#     arima_forecast = model.predict(PREDICT_LEN)
#     m_mase = mase(series, arima_forecast, train_s)
#     m_rmse = rmse(series, arima_forecast)
#     es_mases.append(m_mase)
#     es_rmses.append(m_rmse)
# toc = time.time()
# print("ExponentialSmoothing time elapsed: {}s".format(toc - tic))
# print("ExponentialSmoothing MAPE:", np.mean(es_mases), "+-", np.std(es_mases), 
#         "RMSE:", np.mean(es_rmses), "+-", np.std(es_rmses))

tic = time.time()
theta_mases = []
theta_rmses = []
for c in tqdm(grids):
    c = int(c)
    series = internet_series.univariate_component(c)
    train_s = train.univariate_component(c)
    model = FourTheta(season_mode=SeasonalityMode.ADDITIVE)
    model.fit(train_s)
    arima_forecast = model.predict(PREDICT_LEN)
    m_mase = mase(series, arima_forecast, train_s)
    m_rmse = rmse(series, arima_forecast)
    theta_mases.append(m_mase)
    theta_rmses.append(m_rmse)
toc = time.time()
print("FourTheta time elapsed: {}s".format(toc - tic))
print("FourTheta MAPE:", np.mean(theta_mases), "+-", np.std(theta_mases), 
        "RMSE:", np.mean(theta_rmses), "+-", np.std(theta_rmses))


# tic = time.time()
# arima_mases = []
# arima_rmses = []
# for c in tqdm(grids):
#     c = int(c)
#     series = internet_series.univariate_component(c)
#     train_s = train.univariate_component(c)
#     arima_model = ARIMA()
#     arima_model.fit(train_s)
#     arima_forecast = arima_model.predict(PREDICT_LEN)
#     m_mase = mase(series, arima_forecast, train_s)
#     m_rmse = rmse(series, arima_forecast)
#     arima_mases.append(m_mase)
#     arima_rmses.append(m_rmse)
# toc = time.time()
# print('ARIMA Runing time:', toc - tic)
# print("ARIMA MAPE:", np.mean(arima_mases), "+-", np.std(arima_mases), 
#         "RMSE:", np.mean(arima_rmses), "+-", np.std(arima_rmses))

# tic = time.time()
# naive_mases = []
# naive_rmses = []
# for c in tqdm(grids):
#     c = int(c)
#     series = internet_series.univariate_component(c)
#     seasonal_model = NaiveSeasonal(K=12)
#     seasonal_model.fit(train.univariate_component(c))
#     seasonal_forecast = seasonal_model.predict(PREDICT_LEN)

#     drift_model = NaiveDrift()
#     drift_model.fit(train.univariate_component(c))
#     drift_forecast = drift_model.predict(PREDICT_LEN)

#     combined_forecast = seasonal_forecast + drift_forecast - train.univariate_component(c).last_value()
#     m_mase = mase(series, combined_forecast, train.univariate_component(c))
#     m_rmse = rmse(series, combined_forecast)
#     naive_mases.append(m_mase)
#     naive_rmses.append(m_rmse)
# toc = time.time()
# print('Naive Runing time:', toc - tic)
# print("NAIVE MAPE:", np.mean(naive_mases), "+-", np.std(naive_mases), 
#         "RMSE:", np.mean(naive_rmses), "+-", np.std(naive_rmses))

# tic = time.time()
# nbeats_mases = []
# nbeats_rmses = []
# for c in tqdm(grids):
#     c = int(c)
#     series = internet_series.univariate_component(c)
#     nbeats_model = NBEATSModel(
#         input_chunk_length=24, 
#         output_chunk_length=12,
#         random_state=42,
#         generic_architecture=False,
#         num_blocks=3,
#         num_layers=4,
#         layer_widths=512,
#         n_epochs=100,
#         log_tensorboard=True,
#         show_warnings=False,
#         pl_trainer_kwargs={
#             "accelerator": "gpu", 
#             "gpus": [0],
#             "enable_model_summary": False,
#             "enable_progress_bar": False,
#         }
#     )
#     nbeats_model.fit(train.univariate_component(c), epochs=100, verbose=False)
#     nbeats_forecast = nbeats_model.predict(PREDICT_LEN, verbose=False)
#     m_mase = mase(series, nbeats_forecast, train.univariate_component(c))
#     m_rmse = rmse(series, nbeats_forecast)
#     nbeats_mases.append(m_mase)
#     nbeats_rmses.append(m_rmse)
# toc = time.time()
# print('NBEATS Runing time:', toc - tic)
# print("NBEATS MAPE:", np.mean(nbeats_mases), "+-", np.std(nbeats_mases), "RMSE:", np.mean(nbeats_rmses), "+-", np.std(nbeats_rmses))