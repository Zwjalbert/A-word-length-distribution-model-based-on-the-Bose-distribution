# -*- coding: utf-8 -*-
"""
================================================================
error_analysis.py —— 残差分析与误差诊断
================================================================

提供拟合残差的诊断工具：
    - 标准化残差（皮尔逊残差）的计算与异常点识别
    - 残差自相关 / 随机性的简单检验
    - 卡方贡献分解（定位拟合最差的词长 bin）
    - 拟合稳定性摘要

与 metrics.py 的分工：
    metrics.py 给出全局标量指标；
    error_analysis.py 给出逐 bin 的残差结构与诊断结论。
"""

import numpy as np

# 跨目录导入
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in [_os.path.join(_ROOT, "01_语料处理")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


def pearson_residuals(y_obs, y_pred):
    """
    皮尔逊（标准化）残差 = (观测 - 预测) / √预测。
    返回值应近似服从标准正态分布 N(0,1)。
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-12)
    return (y_obs - y_pred) / np.sqrt(y_pred)


def identify_outliers(residuals, threshold=2.0):
    """
    识别标准化残差绝对值超过 threshold（默认 2σ）的异常 bin。

    返回：
        list[int]: 异常 bin 的下标列表
    """
    residuals = np.asarray(residuals, dtype=float)
    return list(np.where(np.abs(residuals) > threshold)[0])


def chi2_contributions(y_obs, y_pred):
    """
    逐 bin 卡方贡献 = (观测 - 预测)^2 / 预测。
    用于定位对总体卡方贡献最大（拟合最差）的词长。
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-12)
    return (y_obs - y_pred) ** 2 / y_pred


def residual_summary(ell, y_obs, y_pred):
    """
    生成残差诊断摘要字典。

    返回：
        dict:
            pearson_residuals: 标准化残差数组
            n_outliers_2sigma: |残差|>2 的 bin 数
            outlier_indices:    异常 bin 下标
            outlier_lengths:    异常 bin 对应词长
            worst_bin:          卡方贡献最大的 bin
            mean_residual:      平均残差（应为 ~0）
            std_residual:       残差标准差
    """
    ell = np.asarray(ell, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    res = pearson_residuals(y_obs, y_pred)
    outliers = identify_outliers(res, threshold=2.0)
    contrib = chi2_contributions(y_obs, y_pred)
    worst = int(np.argmax(contrib))

    return {
        "pearson_residuals": res.tolist(),
        "n_outliers_2sigma": len(outliers),
        "outlier_indices": outliers,
        "outlier_lengths": [float(ell[i]) for i in outliers],
        "worst_bin": worst,
        "worst_length": float(ell[worst]),
        "mean_residual": float(np.mean(y_obs - y_pred)),
        "std_residual": float(np.std(res)),
    }


def print_residual_report(ell, y_obs, y_pred, model_name="模型"):
    """
    在控制台打印简洁的残差诊断报告。
    """
    s = residual_summary(ell, y_obs, y_pred)
    print(f"--- 残差诊断: {model_name} ---")
    print(f"  平均残差      = {s['mean_residual']:.2f}")
    print(f"  残差标准差    = {s['std_residual']:.3f} (期望≈1)")
    print(f"  |残差|>2σ bin = {s['n_outliers_2sigma']}/{len(ell)}")
    print(f"  最差拟合词长  = ℓ={s['worst_length']:.0f}")
    return s


if __name__ == "__main__":
    ell_demo = np.arange(1, 9, dtype=float)
    obs = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    pred = np.array([160000, 52000, 22000, 9000, 2500, 330, 30, 7], dtype=float)
    print_residual_report(ell_demo, obs, pred, "演示")
