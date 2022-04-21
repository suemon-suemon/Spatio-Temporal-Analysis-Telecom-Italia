def get_indexes_of_train(format, time_level, out_start_idx, close_len, period_len, trend_len = 0):
    if time_level == 'hour':
        TIME_STEPS_OF_DAY = 24
    else: # 10 mins level
        TIME_STEPS_OF_DAY = 24 * 6
    indices = []
    if format == 'default':
        indices += [out_start_idx-i-1 for i in range(close_len)]
        if period_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY-1 for i in range(period_len)]
        if trend_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY*7 for i in range(trend_len)]
    elif format == 'sttran':
        if period_len > 0:
            indices += [out_start_idx-(i+1)*TIME_STEPS_OF_DAY-j for j in range(close_len) for i in range(period_len)]
    indices.reverse()
    return indices