import numpy as np

from sklearn.metrics import mean_squared_error

# calculate NRMSE between arrays of shape (n_samples, n_grids)
def nrmse(y_true, y_pred):
    rmse = [mean_squared_error(y_true[:, i], y_pred[:, i], squared=False) for i in range(y_true.shape[1])]
    return np.mean(rmse / np.mean(y_true, axis=0)).item()

# if __name__ == '__main__':
#     y_true = np.array([[1, 2], [4, 5], [7, 8]])
#     y_pred = np.array([[1, 2], [4, 5], [8, 8]])
#     print(nrmse(y_true, y_pred))