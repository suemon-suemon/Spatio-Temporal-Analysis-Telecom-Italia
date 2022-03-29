from datasets.MilanFG import MilanFullGridDataset
import numpy as np

def test_milan_fullgrid_dataset():
    fg_test_data = np.array([
        [[2, 3, 4],
         [3, 4, 5],
         [4, 5, 6]],

        [[3, 2, 1],
         [0, 3, 4],
         [6, 5, 4]],

        [[2, 4, 0],
         [0, 5, 3],
         [3, 4, 3]],

        [[1, 2, 1],
         [0, 3, 1],
         [2, 3, 10]]
    ])
    test_ds = MilanFullGridDataset(fg_test_data, aggr_time=None, close_len=2)
    assert len(test_ds) == 2
    np.testing.assert_array_equal(test_ds[0][0], fg_test_data[0:2])
    np.testing.assert_array_equal(test_ds[1][1], fg_test_data[3])

    len_ts = 168 * 4 # 4 weeks, 168 = 24 * 7
    close_len = period_len = trend_len = 3
    fg_test_data = np.array([[i] * 9 for i in range(len_ts)]).reshape((len_ts, 3, 3))
    test_ds = MilanFullGridDataset(fg_test_data, aggr_time='hour', close_len=close_len, period_len=period_len, trend_len=trend_len)
    assert len(test_ds) == len_ts - close_len
    test_idx = 169
    assert test_ds[test_idx][0].shape == (9, 3, 3)
    # test_ds[test_idx][0] --> close 0, 1, 2 | period 3, 4, 5 | trend 6, 7, 8
    np.testing.assert_array_equal(test_ds[test_idx][1], np.array([test_idx + close_len] * 9).reshape((3, 3)))
    np.testing.assert_array_equal(test_ds[test_idx][0][2], np.array([test_idx + 2] * 9).reshape((3, 3)))
    np.testing.assert_array_equal(test_ds[test_idx][0][4], np.array([test_idx+3 - 2*24] * 9).reshape((3, 3)))
    np.testing.assert_array_equal(test_ds[test_idx][0][8], np.array([test_idx+3 - 24*7] * 9).reshape((3, 3)))
    np.testing.assert_array_equal(test_ds[test_idx][0][6], np.zeros((3, 3)))

# if __name__ == '__main__':
#     test_milan_fullgrid_dataset()