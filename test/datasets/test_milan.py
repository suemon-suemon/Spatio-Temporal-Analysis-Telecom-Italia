import numpy as np
import pytest

from datasets import MilanSW
from datasets.MilanSW import MilanSlidingWindowDataset


# @pytest.mark.skip(reason="Take too much time")
def test_milan_setup():
    milan_dataset = MilanSW(data_dir="data/sms-call-internet-mi")
    milan_dataset.prepare_data()
    milan_dataset.setup()
    train_dataloader = milan_dataset.train_dataloader()
    for train_data in train_dataloader:
        assert(train_data[0].shape[1] == 12)
        assert(train_data[0].shape[2] == 121)
        # print(train_data[0].shape)
        # print(train_data[1])
        break

def test_milan_dataset():
    test_milan_data = np.array([
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
    test_dataset = MilanSlidingWindowDataset(test_milan_data, window_size=3, input_len=2)
    assert len(test_dataset) == 18
    np.testing.assert_array_equal(test_dataset[16][0], np.array([
        [0, 3, 4, 6, 5, 4 ,0, 0, 0],
        [0, 5, 3, 3, 4, 3, 0, 0, 0]
    ]))
    assert(test_dataset[17][1] == 10)


# if __name__ == '__main__':
#     test_milan_setup()
