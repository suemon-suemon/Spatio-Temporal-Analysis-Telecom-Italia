from fix_path import fix_python_path_if_working_locally

fix_python_path_if_working_locally()

import os
import time

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.metrics import mase, rmse
from darts.models import ARIMA, FourTheta, NaiveDrift, NaiveSeasonal
from darts.utils.utils import SeasonalityMode
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm
from utils.load_data import load_multi_telecom_data
from utils.manipulate_data import (df2cell_time_array, df2timeindex,
                                   filter_grids_data_by_colrow)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
TRAIN_VAL_SPLIT = {'year': 2013, 'month': 11, 'day': 21, 'hour': 0}
PREDICT_LEN = 6 * 24 * 10

# get files from 11-01 to 11-30
paths = ['/home/zzw/dataset/TeleMilan/sms-call-internet-mi/sms-call-internet-mi-2013-11-0{i}.csv'.format(i=i) for i in range(1, 10)] +\
        ['/home/zzw/dataset/TeleMilan/sms-call-internet-mi/sms-call-internet-mi-2013-11-{i}.csv'.format(i=i) for i in range(10, 31)]
data = load_multi_telecom_data(paths)
data = filter_grids_data_by_colrow(data, col1=41, col2=70, row1=41, row2=70)

internet_data = df2cell_time_array(data)
timeindex = pd.DatetimeIndex(df2timeindex(data))
internet_series = TimeSeries.from_times_and_values(times=timeindex, values=internet_data)

scaler = Scaler(MaxAbsScaler())
filler = MissingValuesFiller()
internet_series = scaler.fit_transform(internet_series)
internet_series = filler.transform(internet_series)
train, test = internet_series.split_before(pd.Timestamp(
    year=TRAIN_VAL_SPLIT['year'], 
    month=TRAIN_VAL_SPLIT['month'], 
    day=TRAIN_VAL_SPLIT['day'], 
    hour=TRAIN_VAL_SPLIT['hour'], minute=0, second=0)
)
print("Timesteps of train: {}, test: {}. Total components: {}".format(len(train), len(test), train.n_components))

# grid search parameters
# theta = 2 - np.linspace(-10, 10, 50)
# params_grids = {'theta': theta, 
#           'season_mode': [SeasonalityMode.ADDITIVE], 
#           'model_mode': [ModelMode.ADDITIVE], 
#           'trend_mode': [TrendMode.LINEAR], 
#           'normalization': [False, True]}
# model, params = FourTheta.gridsearch(parameters=params_grids, series=train.univariate_component(585), 
                                    #  forecast_horizon=144, metric=rmse, verbose=True, n_jobs=8)

# arima grid search
params_grids = {'p': [2, 8, 12, 16],
                'd': [1, 2, 4],
                'q': [0, 2, 4, 6],
                'seasonal_order': [(0,0,0,0), (12,1,0,144)]}
model, params = ARIMA.gridsearch(parameters=params_grids, series=train.univariate_component(585), 
                                     forecast_horizon=144, metric=rmse, verbose=True, n_jobs=32)
print(params)

# tic = time.time()
# theta_mases = []
# theta_rmses = []
# for c in tqdm(range(train.n_components)):
#     c = int(c)
#     series = internet_series.univariate_component(c)
#     train_s = train.univariate_component(c)
#     model = FourTheta(theta=0.57, season_mode=SeasonalityMode.ADDITIVE, normalization=False)
#     model.fit(train_s)
#     arima_forecast = model.predict(PREDICT_LEN)
#     m_mase = mase(series, arima_forecast, train_s)
#     m_rmse = rmse(series, arima_forecast)
#     theta_mases.append(m_mase)
#     theta_rmses.append(m_rmse)
# toc = time.time()
# print("FourTheta time elapsed: {}s".format(toc - tic))
# print("FourTheta MAPE:", np.mean(theta_mases), "+-", np.std(theta_mases), 
#         "RMSE:", np.mean(theta_rmses), "+-", np.std(theta_rmses))
