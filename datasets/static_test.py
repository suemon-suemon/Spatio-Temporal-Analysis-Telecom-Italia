import numpy as np
import pandas as pd
import statsmodels.api as sm
from datasets.Milan import MilanDataset
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import linear_reset, acorr_ljungbox
import wandb
import warnings

warnings.filterwarnings('ignore')  # 忽略警告

# ========== Hurst指数计算 ==========
def compute_hurst(ts):
    N = len(ts)
    if N < 100:
        return np.nan  # 太短不稳定
    ts = np.array(ts)
    mean_ts = np.mean(ts)
    Z = np.cumsum(ts - mean_ts)
    R = np.max(Z) - np.min(Z)
    S = np.std(ts)
    if S == 0:
        return np.nan
    return np.log(R/S) / np.log(N)

# ========== DFA计算 ==========
def compute_dfa(ts, min_window=4, max_window=50):
    ts = np.array(ts)
    n = len(ts)
    if n < max_window:
        return np.nan  # 太短不能做dfa
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=10, dtype=int))
    fluctuations = []

    for w in window_sizes:
        segments = n // w
        rms = []
        for i in range(segments):
            segment = ts[i*w:(i+1)*w]
            x = np.arange(w)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuations.append(np.mean(rms))

    try:
        coeffs = np.polyfit(np.log(window_sizes), np.log(fluctuations), 1)
        return coeffs[0]  # DFA scaling exponent
    except:
        return np.nan

# ========== 单节点分析 ==========
def analyze_node_series(ts, node_id):
    result = {"Node": node_id}

    # ====== 线性性检验（RESET） ======
    ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
    X = sm.add_constant(np.arange(len(ts)))
    try:
        model = sm.OLS(ts_norm, X).fit()
        reset_pvalue = linear_reset(model, power=2, use_f=True).pvalue
    except:
        reset_pvalue = np.nan
    result["RESET_pvalue"] = reset_pvalue
    result["Linearity"] = "Linear" if reset_pvalue > 0.05 else "Nonlinear"

    # ====== 平稳性检验（ADF + KPSS） ======
    try:
        adf_pvalue = adfuller(ts, autolag='AIC')[1]
    except:
        adf_pvalue = np.nan
    result["ADF_pvalue"] = adf_pvalue
    result["ADF_Stationary"] = "Stationary" if adf_pvalue < 0.05 else "Non-stationary"

    try:
        kpss_pvalue = kpss(ts, regression='c')[1]
    except:
        kpss_pvalue = np.nan
    result["KPSS_pvalue"] = kpss_pvalue
    result["KPSS_Stationary"] = "Stationary" if kpss_pvalue > 0.05 else "Non-stationary"

    # ====== 短期记忆性检验（Ljung-Box） ======
    try:
        lb_pvalue = acorr_ljungbox(ts, lags=[10], return_df=True)["lb_pvalue"].values[0]
    except:
        lb_pvalue = np.nan
    result["LjungBox_pvalue_lag10"] = lb_pvalue
    result["Short_Memory"] = "Short-Term" if lb_pvalue < 0.05 else "Weak Correlation"

    # ====== 长期记忆性检验（Hurst） ======
    hurst_exp = compute_hurst(ts)
    result["Hurst_Exponent"] = hurst_exp
    if not np.isnan(hurst_exp):
        if hurst_exp < 0.4:
            result["Long_Memory_Hurst"] = "Anti-persistent"
        elif hurst_exp > 0.6:
            result["Long_Memory_Hurst"] = "Persistent"
        else:
            result["Long_Memory_Hurst"] = "Random Walk"
    else:
        result["Long_Memory_Hurst"] = "Undefined"

    # ====== 长期记忆性检验（DFA） ======
    dfa_exp = compute_dfa(ts)
    result["DFA_Exponent"] = dfa_exp

    return result

# ========== 多节点分析 ==========
def analyze_all_nodes(X, use_wandb=False):
    T, N = X.shape
    all_results = []

    for node in range(N):
        print(f"Analyzing node {node}...")
        ts = X[:, node]
        result = analyze_node_series(ts, node)
        all_results.append(result)

    df_results = pd.DataFrame(all_results)

    if use_wandb:
        # 上传到 wandb
        table = wandb.Table(columns=df_results.columns.tolist())
        for _, row in df_results.iterrows():
            table.add_data(*row.values.tolist())
        wandb.log({"Node Statistical Tests": table})

    return df_results

# ====== 示例使用 ======
if __name__ == "__main__":
    # 数据是 X, shape [T, N]

    dataset = MilanDataset(aggr_time='10min',
                    time_range='all',
                    # grid_range=p['grid_range'],
                    load_meta=False,
                    normalize=False,
                    )
    dataset.prepare_data()
    dataset.setup()
    X = dataset.milan_grid_data.squeeze().reshape(-1, dataset.N_all)

    wandb.init(project="MilanNodeStatTests", name="StatProperty", mode="offline")

    results_df = analyze_all_nodes(X, use_wandb=True)

    wandb.finish()