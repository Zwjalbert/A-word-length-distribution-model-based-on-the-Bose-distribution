# -*- coding: utf-8 -*-
"""
================================================================
metrics.py —— 拟合优度统计指标
================================================================

提供模型拟合效果评价的全套统计量：
    - 决定系数：R², 调整 R² (R²_adj)
    - 尺度误差：RMSE, MAE, MAPE
    - 信息准则：对数似然 log_likelihood, AIC, AICc, BIC
    - 模型相对权重：AIC_weight, BIC_weight
    - 分布检验：卡方 (χ²) 及其 N、自由度、p 值；KS 检验
    - 标准化残差 std_residuals

似然采用计数数据的泊松假设：
    log L = Σ [ y_i ln(ŷ_i) - ŷ_i - ln(y_i!) ]
信息准则：
    AIC      = 2k - 2 ln L
    AICc     = AIC + 2k(k+1)/(n-k-1)
    BIC      = k ln n - 2 ln L
    AIC_weight_i = exp(-0.5·ΔAIC_i) / Σ_j exp(-0.5·ΔAIC_j)
"""

import numpy as np
from scipy.stats import chi2, ks_2samp, kstwo

# 跨目录导入 config
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG01 = _os.path.join(_os.path.dirname(_HERE), "01_语料处理")
if _PKG01 not in _sys.path:
    _sys.path.insert(0, _PKG01)
from config import CHI2_MIN_EXPECTED, ENABLE_YATES_CORRECTION  # noqa: E402


# ================================================================
# 1. 连续拟合精度指标
# ================================================================
def r_squared(y_obs, y_pred):
    """普通决定系数 R²。"""
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def adjusted_r_squared(y_obs, y_pred, n_params):
    """调整决定系数 R²_adj = 1 - (1-R²)(n-1)/(n-k-1)。"""
    n = len(y_obs)
    if n - n_params - 1 <= 0:
        return 0.0
    r2 = r_squared(y_obs, y_pred)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)


def rmse(y_obs, y_pred):
    """均方根误差 RMSE。"""
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


def mae(y_obs, y_pred):
    """平均绝对误差 MAE。"""
    return float(np.mean(np.abs(y_obs - y_pred)))


def mape(y_obs, y_pred, eps=1e-6):
    """平均绝对百分比误差 MAPE（对 0 观测做截断）。"""
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_obs) > eps
    return float(np.mean(np.abs((y_obs[mask] - y_pred[mask]) / y_obs[mask])) * 100.0) \
        if np.any(mask) else 0.0


# ================================================================
# 2. 对数似然与信息准则
# ================================================================
def poisson_log_likelihood(y_obs, y_pred):
    """
    泊松对数似然：ln L = Σ [ y ln(ŷ) - ŷ - ln(y!) ]。
    预测值截断为正，避免 log(0)。
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)
    return float(np.sum(y_obs * np.log(y_pred) - y_pred -
                        np.vectorize(_logfact)(y_obs)))


def _logfact(y):
    """ln(y!) 用 Stirling 近似，y 可为浮点计数。"""
    from scipy.special import gammaln
    return float(gammaln(y + 1.0))


def information_criteria(y_obs, y_pred, n_params):
    """
    计算 log_likelihood / AIC / AICc / BIC。

    参数：
        y_obs:    观测计数
        y_pred:   预测计数
        n_params: 模型自由参数个数
    """
    n = len(y_obs)
    ll = poisson_log_likelihood(y_obs, y_pred)
    aic = 2.0 * n_params - 2.0 * ll
    # AICc 小样本修正
    denom = n - n_params - 1
    aicc = aic + (2.0 * n_params * (n_params + 1) / denom) if denom > 0 else np.inf
    bic = n_params * np.log(n) - 2.0 * ll
    return {
        "log_likelihood": ll,
        "AIC": float(aic),
        "AICc": float(aicc),
        "BIC": float(bic),
    }


def aic_weights(aic_values):
    """
    计算 AIC 权重：w_i = exp(-0.5·ΔAIC_i) / Σ_j exp(-0.5·ΔAIC_j)。

    参数：
        aic_values: 各模型 AIC 值列表/数组

    返回：
        np.ndarray: 与输入等长的权重数组（和为 1）
    """
    aic_values = np.asarray(aic_values, dtype=float)
    if len(aic_values) == 0:
        return np.array([])
    aic_min = np.nanmin(aic_values)
    delta = aic_values - aic_min
    w = np.exp(-0.5 * delta)
    s = np.sum(w)
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def bic_weights(bic_values):
    """BIC 权重，公式同 AIC 权重。"""
    return aic_weights(np.asarray(bic_values, dtype=float))


# ================================================================
# 3. 卡方拟合优度检验
# ================================================================
def chi2_test(y_obs, y_pred, n_params, min_expected=CHI2_MIN_EXPECTED):
    """
    卡方拟合优度检验（按最小期望频数合并尾部 bin）。

    返回：
        dict: chi2, df, chi2_df, chi2_N, p_value, 评价
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    N_total = np.sum(y_obs)

    obs_combined, pred_combined = [], []
    cur_o = cur_p = 0.0
    for o, p in zip(y_obs, y_pred):
        cur_o += o
        cur_p += p
        if cur_p >= min_expected:
            obs_combined.append(cur_o)
            pred_combined.append(cur_p)
            cur_o = cur_p = 0.0
    if cur_o > 0 or cur_p > 0:
        if len(pred_combined) > 0:
            obs_combined[-1] += cur_o
            pred_combined[-1] += cur_p
        else:
            obs_combined.append(cur_o)
            pred_combined.append(cur_p)

    obs_combined = np.array(obs_combined)
    pred_combined = np.maximum(np.array(pred_combined), 1e-10)

    if ENABLE_YATES_CORRECTION and len(obs_combined) == 2:
        chi2_val = float(np.sum((np.abs(obs_combined - pred_combined) - 0.5) ** 2 / pred_combined))
    else:
        chi2_val = float(np.sum((obs_combined - pred_combined) ** 2 / pred_combined))

    df = max(1, len(obs_combined) - n_params - 1)
    p_value = float(chi2.sf(chi2_val, df))
    chi2_N = chi2_val / N_total if N_total > 0 else np.nan

    if chi2_N < 0.01:
        eval_str = "拟合极优"
    elif chi2_N < 0.05:
        eval_str = "拟合优秀"
    elif chi2_N < 0.10:
        eval_str = "拟合良好"
    elif chi2_N < 0.20:
        eval_str = "拟合一般"
    else:
        eval_str = "拟合较差"

    return {
        "chi2": chi2_val,
        "df": df,
        "chi2_df": chi2_val / df,
        "chi2_N": float(chi2_N),
        "p_value": p_value,
        "chi2_eval": eval_str,
    }


# ================================================================
# 4. KS 检验
# ================================================================
def ks_test(ell, y_obs, y_pred):
    """
    对观测分布与预测分布做 Kolmogorov-Smirnov 检验。

    直接基于计数构造经验 CDF，避免 np.repeat 展开大样本导致内存溢出。
    返回 (KS 统计量, p 值)。
    """
    ell = np.asarray(ell, dtype=float)
    y_obs = np.maximum(np.asarray(y_obs, dtype=float), 0.0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)

    total_obs = y_obs.sum()
    total_pred = y_pred.sum()
    if total_obs < 2 or total_pred < 2:
        return 0.0, 1.0

    # 直接计算 CDF，不展开样本
    cdf_obs = np.cumsum(y_obs) / total_obs
    cdf_pred = np.cumsum(y_pred) / total_pred
    ks_stat = float(np.max(np.abs(cdf_obs - cdf_pred)))

    # 两样本 KS 检验的有效样本量
    n_eff = np.sqrt(total_obs * total_pred / (total_obs + total_pred))
    # 使用 kstwo 分布计算 p 值（两样本 KS 检验的渐近分布）
    try:
        p_value = float(kstwo.sf(ks_stat, int(round(n_eff))))
    except Exception:
        # 兜底：用近似公式
        lam = (n_eff + 0.12 + 0.11 / n_eff) * ks_stat
        p_value = 2.0 * np.exp(-2.0 * lam * lam)
        p_value = float(min(max(p_value, 0.0), 1.0))

    return ks_stat, p_value


# ================================================================
# 5. 一站式汇总
# ================================================================
def compute_all_metrics(ell, y_obs, y_pred, n_params):
    """
    汇总计算全部拟合指标。

    参数：
        ell:      词长数组
        y_obs:    观测计数
        y_pred:   预测计数
        n_params: 自由参数个数

    返回：
        dict: 含 R2, R2_adj, RMSE, MAE, MAPE, log_likelihood,
              AIC, AICc, BIC, chi2*, KS*, std_residuals 等
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_obs - y_pred
    y_err = np.sqrt(np.maximum(y_obs, 1e-12))
    std_residuals = residuals / y_err

    ic = information_criteria(y_obs, y_pred, n_params)
    c2 = chi2_test(y_obs, y_pred, n_params)
    ks_stat, ks_p = ks_test(ell, y_obs, y_pred)

    return {
        "n_data_points": len(y_obs),
        "n_params": n_params,
        "effective_df": len(y_obs) - n_params,
        "R2": float(r_squared(y_obs, y_pred)),
        "R2_adj": float(adjusted_r_squared(y_obs, y_pred, n_params)),
        "RMSE": rmse(y_obs, y_pred),
        "MAE": mae(y_obs, y_pred),
        "MAPE": mape(y_obs, y_pred),
        **ic,
        "chi2": c2["chi2"],
        "chi2_df": c2["chi2_df"],
        "chi2_N": c2["chi2_N"],
        "chi2_pvalue": c2["p_value"],
        "chi2_eval": c2["chi2_eval"],
        "KS_stat": ks_stat,
        "KS_pvalue": ks_p,
        "std_residuals": std_residuals.tolist(),
    }


if __name__ == "__main__":
    # 自检
    ell_demo = np.arange(1, 9, dtype=float)
    obs = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    pred = np.array([160000, 52000, 22000, 9000, 2500, 330, 30, 7], dtype=float)
    m = compute_all_metrics(ell_demo, obs, pred, 2)
    for k, v in m.items():
        if k != "std_residuals":
            print(f"  {k:16s}: {v}")
