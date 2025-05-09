from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from tensorflow.python.framework.errors_impl import UnimplementedError


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def get_time_features(
        timestamps,
        timeenc = 1,
        freq = 'h',
        time_feature_period = True,
        aggr_time='10min',
        holidays = None
):
    """
    通用版时间特征提取。

    参数:
      - timestamps: pd.Series or np.ndarray of datetime
      - timeenc: 是否标准时间编码
      - freq: 时间粒度 'h', 't', 'm'...
      - time_feature_period: 是否使用周期型 sin/cos 编码
      - aggr_time: 'hour', '10min' 等，用于step_of_day计算
      - holidays: 自定义节假日列表

    返回:
      - if timeenc == 0 and freq='h': (T, 4) ndarray, 保留原数值，没有编码
      - if timeenc == 1 and freq='h': (T, 4) ndarray, 每个数值都编码到了[-0.5, 0.5]
      - if timeenc == 2 and time_feature_period == True: (T, 9) ndarray
      - if timeenc == 2 and time_feature_period == False: (T, 6) ndarray

    """
    if isinstance(timestamps, np.ndarray):
        timestamps = pd.to_datetime(timestamps)

    # 基础时间字段
    month = timestamps.month
    day_of_week = timestamps.dayofweek
    hour = timestamps.hour
    minute = timestamps.minute

    if aggr_time == 'hour':
        step_of_day = hour
        steps_per_day = 24
    elif aggr_time == '10min' or aggr_time is None:
        step_of_day = hour * 6 + minute // 10
        steps_per_day = 144
    elif aggr_time == '5min':
        step_of_day = hour * 12 + minute // 5
        steps_per_day = 288
    else:
        raise UnimplementedError('aggr_time is not supported')

    # 判断是否凌晨
    is_midnight = ((hour >= 1) & (hour <= 6)).astype(int)
    is_weekend = (day_of_week >= 5).astype(int)

    # 节假日
    if holidays is not None:
        is_holiday = timestamps.strftime('%Y-%m-%d').isin(holidays).astype(int)
    else:
        is_holiday = np.zeros(len(timestamps), dtype=int)

    if timeenc == 0:
        # 传统简单版
        df = pd.DataFrame({
            'month': month,
            'day': timestamps.day,
            'weekday': day_of_week,
            'hour': hour,
            'minute': (minute // 15),
        })

        freq_map = {
            'y': [],
            'm': ['month'],
            'w': ['month'],
            'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'],
            'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return df[freq_map[freq.lower()]].values

    elif timeenc == 1: # one-hot 编码
        dates = pd.to_datetime(timestamps.values)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)

    elif timeenc == 2:
        if time_feature_period:
            # 周期型（sin/cos编码 + 二进制特征）
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)

            sin_day_of_week = np.sin(2 * np.pi * day_of_week / 7)
            cos_day_of_week = np.cos(2 * np.pi * day_of_week / 7)

            sin_step_of_day = np.sin(2 * np.pi * step_of_day / steps_per_day)
            cos_step_of_day = np.cos(2 * np.pi * step_of_day / steps_per_day)

            return np.stack([
                sin_month, cos_month,
                sin_day_of_week, cos_day_of_week,
                sin_step_of_day, cos_step_of_day,
                is_midnight, is_weekend, is_holiday
            ], axis=-1)  # (T, 9)
        else:
            # 直接数值型时间特征
            return np.stack([
                month, day_of_week, step_of_day,
                is_midnight, is_weekend, is_holiday
            ], axis=-1)  # (T, 6)
    else:
        raise UnimplementedError('timeenc is not supported')


if __name__ == '__main__':
    timestamps = pd.date_range('2023-04-01 00:00', '2023-05-02 00:00', freq='5H')
    # dates = pd.DataFrame({'date': timestamps})
    time_feature = get_time_features(
        timestamps,
        timeenc=0,
        freq='h',
        time_feature_period=True,
        aggr_time='10min',
        holidays=None
    )
    print('time feature shape', time_feature.shape)
    print(time_feature)
