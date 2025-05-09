import numpy as np
import pmdarima as pm
import wandb
from joblib import Parallel, delayed
import time
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from utils.registry import register

@register("arima")
class ARIMAMultiNode:
    def __init__(self,
                 X, # [T, N]
                 input_len: int = 6,
                 pred_len: int = 3,
                 train_len: int = 1000,
                 val_len: int = 100,
                 logger = None,
                 is_period = True, # 是否考虑周期性
                 period_len = None, # 一个周期多少点
                 ):
        """
        参数：
        X: 数据矩阵 [T, N]
        input_len: 使用多少步时间作为输入（默认6）
        pred_len: 预测未来多少步（默认3）

        模型：多节点的ARIMA模型，阶数自动选择，周期性可选。
             使用[:train_len]的数据拟合模型，使用[train_len+val_len:]数据测试性能。
             测试由 input_len 预测 pred_len 的性能，记录打印每个节点的最佳阶数和预测性能

        ARIMA(p,d,q) 由三部分组成：
        	•	AR(p)：自回归部分；
        	•	I(d)：差分部分（积分）；
        	•	MA(q)：滑动平均部分
            •	ARIMA(0,0,q) ==> 滑动平均模型

        自动选择最佳阶数的原理是：
            1. 对 d（差分阶数）进行单位根检验网格遍历
            2. 对 (p,q)（AR、MA 阶数）进行网格遍历
            3. 比较每一组模型的 AIC/BIC
            4. 选出 AIC 最低的模型作为最终结果

        当考虑周期性时, 拟合更大的 SARIMA 模型 SARIMA(p,d,q)(P,D,Q)[m]：
            1.	先季节性差分 (1-L^m)^D y_t，让序列去除周期性趋势
            2.	再普通差分 (1-L)^d，让序列平稳
            3.	建立SARIMA模型，同时拟合
                •	非季节部分 p, q；
                •	季节部分 P, Q。
            4.	一起搜索最优组合：
                •	(p,d,q) × (P,D,Q)
        """
        self.X = X
        self.T, self.N = self.X.shape
        self.input_len = input_len
        self.pred_len = pred_len
        self.train_end = train_len
        self.test_start = train_len + val_len
        self.is_period = is_period
        self.period_len = None
        if self.is_period:
            self.period_len = period_len

        self.logger = logger  # wandb logger 传入

    def fit_arima_model(self, train_series):
        """
        对单个节点拟合ARIMA模型
        """
        try:
            model = pm.auto_arima(train_series,
                                  start_p=0, start_q=1, d=0, # 搜索阶数起点
                                  # max_p=0, max_q=8,  # 搜索最大阶数
                                  seasonal = self.is_period, # 默认允许周期性建模
                                  m = self.period_len, # 周期长度
                                  start_P=0, start_Q=0, D=1, # 周期搜索阶数
                                  stepwise=True,  # 启用贪心搜索加速
                                  suppress_warnings=True,
                                  error_action='ignore')
            return model
        except:
            return None

    def predict_with_model(self, model, ts, train_end, test_start):
        """
        用ARIMA模型在测试区间做滑窗预测
        :param model: 拟合好的ARIMA模型
        :param ts: 该节点的时间序列 [T]
        :return: preds, gts → shape: [num_samples, pred_len]
        """
        T = len(ts)
        preds, gts = [], []

        for t in range(test_start, T - self.input_len - self.pred_len + 1):
            true_y = ts[t + self.input_len : t + self.input_len + self.pred_len]

            if model:
                try:
                    forecast = model.predict(n_periods=(t + self.input_len + self.pred_len - train_end))[-self.pred_len:]
                except:
                    forecast = np.full(self.pred_len, ts[train_end - 1])
            else:
                forecast = np.full(self.pred_len, ts[train_end - 1])

            preds.append(forecast)
            gts.append(true_y)

        return np.array(preds), np.array(gts)

    def evaluate(self, gts, preds):
        """
        输出评估指标：MAE, MAPE, R2, RMSE
        """
        print('start evaluating...')
        gt_flat = gts.ravel()
        pred_flat = preds.ravel()

        mae = mean_absolute_error(gt_flat, pred_flat)
        mape = mean_absolute_percentage_error(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)
        rmse = np.mean([
            mean_squared_error(gts[i].flatten(), preds[i].flatten(), squared=False)
            for i in range(gts.shape[0])
        ])
        return mae, mape, r2, rmse

    def fit_and_predict_node(self, node):
        ts = self.X[:, node]
        train_series = ts[:self.train_end]

        start_time = time.time()
        model = self.fit_arima_model(train_series)
        end_time = time.time()
        train_time = end_time - start_time

        ar, i, ma = (-1, -1, -1)
        if model:
            ar, i, ma = model.order

        preds, gts = self.predict_with_model(model, ts, self.train_end, self.test_start)

        # 单节点评估
        gt_flat = gts.ravel()
        pred_flat = preds.ravel()
        node_mae = mean_absolute_error(gt_flat, pred_flat)
        node_mape = mean_absolute_percentage_error(gt_flat, pred_flat)
        node_r2 = r2_score(gt_flat, pred_flat)
        node_rmse = mean_squared_error(gt_flat, pred_flat, squared=False)

        print(
            f'node: {node}, AR: {ar}, I: {i}, MA: {ma}, train time: {train_time:.4f}s, MAE: {node_mae:.4f}, MAPE: {node_mape:.4f}')

        return node, ar, i, ma, train_time, node_mae, node_mape, node_r2, node_rmse, preds, gts

    def run(self):
        if self.logger:
            per_node_table = wandb.Table(columns=["Node", "AR", "I", "MA", "Train Time (s)", "Node MAE", "Node MAPE", "Node R2", "Node RMSE"])
        else:
            per_node_table = None

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(self.fit_and_predict_node)(node) for node in range(self.N)
        )

        all_preds, all_gts = [], []

        for node, ar, i, ma, train_time, node_mae, node_mape, node_r2, node_rmse, preds, gts in results:
            all_preds.append(preds)
            all_gts.append(gts)

            if self.logger:
                per_node_table.add_data(node, ar, i, ma, train_time, node_mae, node_mape, node_r2, node_rmse)

        preds_arr = np.array(all_preds)  # shape: [N, num_samples, pred_len]
        gts_arr = np.array(all_gts)

        mae, mape, r2, rmse = self.evaluate(gts_arr, preds_arr)

        print("========== ARIMA Baseline Results ==========")
        print(f"Input Length     : {self.input_len}")
        print(f"Prediction Length: {self.pred_len}")
        print(f"MAE              : {mae:.4f}")
        print(f"MAPE             : {mape:.4f}")
        print(f"R2               : {r2:.4f}")
        print(f"RMSE             : {rmse:.4f}")

        # ✅ wandb logging
        if isinstance(self.logger, WandbLogger):
            table = wandb.Table(columns=["Input Len", "Pred Len", "MAE", "MAPE", "R2", "RMSE"])
            table.add_data(self.input_len, self.pred_len, mae, mape, r2, rmse)
            self.logger.experiment.log({
                "Results Table": table,
                "Per Node Table": per_node_table
            })

        return mae, mape, r2, rmse


class LocalARIMAMultiNode:
    def __init__(self,
                 X,
                 input_len: int = 6,
                 pred_len: int = 3,
                 logger = None,
                 ):
        """
        局部滑动窗口ARIMA基准模型
        :param X: 输入数据 [T, N]
        :param input_len: 输入序列长度
        :param pred_len: 预测序列长度
        :param logger: wandb logger
        :param arima_mode: "local" 每个小窗口局部训练一个ARIMA
        """
        self.X = X
        self.T, self.N = self.X.shape
        self.input_len = input_len
        self.pred_len = pred_len
        self.logger = logger

    def fit_and_predict_node(self, node):
        """
        针对单个节点，进行滑动窗口局部ARIMA拟合和预测
        """
        ts = self.X[:, node]
        preds_list = []
        gts_list = []
        start_time = time.time()

        for t in range(0, self.T - self.input_len - self.pred_len + 1):
            train_x = ts[t: t + self.input_len]
            true_y = ts[t + self.input_len: t + self.input_len + self.pred_len]

            # 每个小窗口局部训练
            try:
                model = pm.auto_arima(
                    train_x,
                    start_p=0, max_p=0,
                    start_q=1, max_q=5,
                    d=0,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                pred_y = model.predict(n_periods=self.pred_len)
            except:
                pred_y = np.full(self.pred_len, train_x[-1])  # fallback为最后一个值

            preds_list.append(pred_y)
            gts_list.append(true_y)

        end_time = time.time()
        elapsed_time = end_time - start_time

        preds_arr = np.array(preds_list)  # shape: [num_samples, pred_len]
        gts_arr = np.array(gts_list)

        # 每个节点计算平均指标
        mae = mean_absolute_error(gts_arr.flatten(), preds_arr.flatten())
        mape = mean_absolute_percentage_error(gts_arr.flatten(), preds_arr.flatten())
        mse = mean_squared_error(gts_arr.flatten(), preds_arr.flatten())

        print(f"Node {node} done. MAE: {mae:.4f}, MAPE: {mape:.4f}, MSE: {mse:.4f}, Time: {elapsed_time:.2f}s")

        return node, mae, mape, mse, elapsed_time

    def run(self):
        """
        主流程：
        - 每个节点滑动窗口局部拟合
        - 评估整体结果
        """

        # ✅ 使用多进程加速
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.fit_and_predict_node)(node) for node in range(self.N)
        )

        # 汇总结果
        all_mae, all_mape, all_mse, all_time = [], [], [], []

        if self.logger:
            per_node_table = wandb.Table(columns=["Node", "MAE", "MAPE", "MSE", "Time"])
        else:
            per_node_table = None

        for node, mae, mape, mse, train_time in results:
            all_mae.append(mae)
            all_mape.append(mape)
            all_mse.append(mse)
            all_time.append(train_time)

            if isinstance(self.logger, WandbLogger):
                per_node_table.add_data(node, mae, mape, mse, train_time)

        # 统计整体平均
        avg_mae = np.mean(all_mae)
        avg_mape = np.mean(all_mape)
        avg_mse = np.mean(all_mse)

        print("========== Sliding Window ARIMA Baseline ==========")
        print(f"Input Length     : {self.input_len}")
        print(f"Prediction Length: {self.pred_len}")
        print(f"Avg MAE          : {avg_mae:.4f}")
        print(f"Avg MAPE         : {avg_mape:.4f}")
        print(f"Avg MSE          : {avg_mse:.4f}")

        # ✅ wandb logging
        if isinstance(self.logger, WandbLogger):
            table = wandb.Table(columns=["Input Len", "Pred Len", "MAE", "MAPE", "MSE"])
            table.add_data(self.input_len, self.pred_len, avg_mae, avg_mape, avg_mse)
            self.logger.experiment.log({
                "Results Table": table,
                "Per Node Table": per_node_table
            })
        return avg_mae, avg_mape, avg_mse
