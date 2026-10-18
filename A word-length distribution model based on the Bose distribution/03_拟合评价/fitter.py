# -*- coding: utf-8 -*-
"""
================================================================
fitter.py —— 模型拟合引擎
================================================================

对 scipy.optimize.curve_fit 的封装，提供：
    - 参数边界与多初始值（multi-start）策略
    - 拟合失败的优雅降级
    - 参数标准误与 95% 置信区间估计
    - 统一的拟合结果结构化输出

设计要点：
    - 标准模型（fit_scale=False）：scale 固定为总计数，仅拟合形状参数。
    - 玻色子模型（fit_scale=True）：scale 作为自由参数一并拟合。
"""

import numpy as np
from scipy.optimize import curve_fit

# 跨目录导入
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in [_os.path.join(_ROOT, "01_语料处理"),
           _os.path.join(_ROOT, "02_模型定义")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import (FIT_MAXFEV, FIT_FTOL, ENABLE_MULTI_INIT,
                    MULTI_INIT_NUM, RANDOM_SEED)  # noqa: E402
from models import MODEL_REGISTRY, bose_model  # noqa: E402


# ================================================================
# 各模型默认初始值与边界（基于先验实验结果）
# ================================================================
# 顺序与 MODEL_REGISTRY[model_id]["shape_params"] 一致
DEFAULT_P0 = {
    "shifted_poisson": [1.5],
    "gamma": [2.0, 0.5],
    "poisson_mixture": [0.7, 0.5, 2.5],
    "zipf_alekseev": [0.0, 2.0],
    "negative_binomial": [2.0, 0.5],
    "hyper_poisson": [1.5, 1.0],
    "log_normal": [0.5, 0.8],
    "bose": [3.5, 1.4, 1.0, 1e5],
    "ehp": [1.5, 1.0, 0.5],
    "generalized_gamma": [2.0, 3.0, 1.0],
    "sichel": [1.0, 1.0, -2.0],
}

DEFAULT_BOUNDS = {
    "shifted_poisson": ([1e-3], [50.0]),
    "gamma": ([1e-3, 1e-3], [50.0, 50.0]),
    "poisson_mixture": ([0.0, 1e-3, 1e-3], [1.0, 20.0, 20.0]),
    "zipf_alekseev": ([-10.0, 0.1], [10.0, 10.0]),
    "negative_binomial": ([1e-3, 1e-3], [50.0, 0.9999]),
    "hyper_poisson": ([1e-3, 1e-3], [50.0, 50.0]),
    "log_normal": ([-5.0, 1e-3], [5.0, 10.0]),
    "bose": ([0.1, 1e-3, 0.0, 1.0], [15.0, 8.0, 5.0, 1e9]),
    "ehp": ([1e-3, 1e-3, 1e-3], [50.0, 50.0, 5.0]),
    "generalized_gamma": ([1e-3, 1e-3, 1e-3], [50.0, 50.0, 50.0]),
    "sichel": ([1e-3, 1e-6, -20.0], [20.0, 10.0, 10.0]),
}


def _build_model_func(model_id, total_count, dist_type):
    """
    构造供 curve_fit 使用的函数 f(ell, *params)。
    标准模型：params = 形状参数；内部乘固定 scale=total_count。
    玻色子：params = (d, beta, mu_tilde, scale)。
    """
    spec = MODEL_REGISTRY[model_id]

    if model_id == "bose":
        def f(ell, d, beta, mu_tilde, scale):
            return bose_model(ell, d, beta, mu_tilde, scale, dist_type=dist_type)
        return f
    else:
        pmf = spec["pmf"]
        shape_names = spec["shape_params"]

        def f(ell, *shape):
            return total_count * pmf(ell, *shape)
        f.__name__ = f"f_{model_id}"
        return f


def _generate_starts(model_id, p0_base, bounds):
    """
    生成多组初始值：基准值 + 少量扰动/随机值，避免局部极小。
    """
    starts = [np.array(p0_base, dtype=float)]
    if not ENABLE_MULTI_INIT:
        return starts

    rng = np.random.default_rng(RANDOM_SEED)
    lo, hi = bounds
    lo, hi = np.array(lo), np.array(hi)
    for _ in range(MULTI_INIT_NUM):
        # 在边界内对数/线性随机采样
        frac = rng.uniform(0.1, 0.9, size=len(p0_base))
        s = lo + frac * (hi - lo)
        starts.append(s)
    return starts


def fit_model(model_id, ell, y_obs, total_count, dist_type="token",
              p0=None, bounds=None, sigma=None):
    """
    拟合一单个模型，返回结构化结果。

    参数：
        model_id:    模型标识
        ell:         词长数组
        y_obs:       观测计数
        total_count: 总计数（标准模型用作固定 scale）
        dist_type:   token/type（玻色子模型用）
        p0:          初始值（None 用默认）
        bounds:      参数边界（None 用默认）
        sigma:       观测误差（可选，用于加权最小二乘）

    返回：
        dict: 拟合结果（见 _package_result）；失败时 fit_success=False
    """
    spec = MODEL_REGISTRY[model_id]
    shape_names = spec["shape_params"]
    n_params = len(shape_names)
    rng = np.random.default_rng(RANDOM_SEED)

    p0_base = p0 if p0 is not None else DEFAULT_P0[model_id]
    bounds = bounds if bounds is not None else DEFAULT_BOUNDS[model_id]

    f = _build_model_func(model_id, total_count, dist_type)

    # 玻色子模型：精心选择4组初始值，避免局部最优
    if model_id == "bose" and p0 is None:
        p1_ratio = float(y_obs[0]) / float(total_count) if total_count > 0 else 0.5
        if p1_ratio > 0.80:
            beta_mid, beta_lo, beta_hi = 1.9, 1.6, 2.2
        elif p1_ratio > 0.70:
            beta_mid, beta_lo, beta_hi = 1.6, 1.3, 1.9
        elif p1_ratio > 0.60:
            beta_mid, beta_lo, beta_hi = 1.2, 0.9, 1.5
        else:
            beta_mid, beta_lo, beta_hi = 1.0, 0.7, 1.3
        scale_mid = total_count * 0.3
        starts = [
            np.array([2.8, beta_mid, 0.0, scale_mid], dtype=float),
            np.array([2.8, beta_lo, 0.0, scale_mid], dtype=float),
            np.array([2.8, beta_hi, 0.0, scale_mid], dtype=float),
            np.array([2.8, beta_mid, 0.1, scale_mid], dtype=float),
        ]
    else:
        starts = _generate_starts(model_id, p0_base, bounds)

    best = None
    best_cost = np.inf
    last_err = None

    for s0 in starts:
        try:
            kw = dict(p0=s0, bounds=bounds, maxfev=FIT_MAXFEV, ftol=FIT_FTOL)
            if sigma is not None:
                kw["sigma"] = sigma
                kw["absolute_sigma"] = True
            popt, pcov = curve_fit(f, ell, y_obs, **kw)
            y_pred = f(ell, *popt)
            cost = float(np.sum((y_obs - y_pred) ** 2 /
                                np.maximum(y_obs, 1.0)))
            if cost < best_cost:
                best_cost = cost
                best = (popt, pcov)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    if best is None:
        return {
            "model_id": model_id,
            "fit_success": False,
            "error": str(last_err) if last_err else "所有初始值均失败",
            "predicted": np.zeros_like(ell).tolist(),
            "observed": np.asarray(y_obs).tolist(),
            "lengths": np.asarray(ell).tolist(),
            "n_params": n_params,
            "param_names": shape_names,
            "params": None,
            "param_dict": {},
        }

    popt, pcov = best
    y_pred = f(ell, *popt)
    return _package_result(model_id, shape_names, popt, pcov,
                           ell, y_obs, y_pred, n_params)


def _package_result(model_id, shape_names, popt, pcov,
                    ell, y_obs, y_pred, n_params):
    """打包拟合结果：参数字典、标准误、置信区间。"""
    std_err = np.sqrt(np.maximum(np.diag(pcov), 0.0)) if pcov is not None \
        else np.zeros_like(popt)
    z = 1.96
    ci_low = popt - z * std_err
    ci_high = popt + z * std_err

    param_dict = {name: float(popt[i]) for i, name in enumerate(shape_names)}
    return {
        "model_id": model_id,
        "fit_success": True,
        "lengths": np.asarray(ell).tolist(),
        "observed": np.asarray(y_obs).tolist(),
        "predicted": np.asarray(y_pred).tolist(),
        "n_params": n_params,
        "param_names": list(shape_names),
        "params": np.asarray(popt).tolist(),
        "param_dict": param_dict,
        "param_se": std_err.tolist(),
        "param_ci": np.column_stack([ci_low, ci_high]).tolist(),
    }


if __name__ == "__main__":
    # 自检：用合成数据拟合 shifted_poisson
    ell_demo = np.arange(1, 9, dtype=float)
    obs = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    res = fit_model("shifted_poisson", ell_demo, obs, total_count=obs.sum())
    print("fit_success:", res["fit_success"])
    print("param_dict :", res.get("param_dict"))
