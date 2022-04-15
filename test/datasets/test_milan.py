import numpy as np

from datasets.Milan import Milan


def test_milan_setup():
    milan = Milan(tele_column='mobile', time_range='30days')
    milan.prepare_data()
    milan.setup()
    data = milan.milan_grid_data
    print(np.any(np.isnan(data)), np.all(np.isfinite(data)))
    assert not np.any(np.isnan(data))
    assert np.all(np.isfinite(data))

if __name__ == '__main__':
    test_milan_setup()
