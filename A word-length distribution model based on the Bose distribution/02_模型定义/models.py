# -*- coding: utf-8 -*-
"""
================================================================
models.py —— 11 个词长分布概率质量函数（PMF）定义
================================================================

本模块定义用于词长分布拟合的全部 11 个候选模型。每个模型提供：
    - 归一化的概率质量函数 pmf(ell, *shape_params)
    - 预测计数函数 model(ell, *shape_params, scale) = scale * pmf(ell)

模型清单（与 config.MODEL_META 对应）：
    1.  shifted_poisson     1-位移泊松        1 参数 lambda
    2.  gamma               Gamma             2 参数 alpha, beta
    3.  poisson_mixture     混合泊松          3 参数 pi, lambda1, lambda2
    4.  zipf_alekseev       Zipf-Alekseev     2 参数 a, b
    5.  negative_binomial  负二项            2 参数 r, p
    6.  hyper_poisson       Hyper-Poisson     2 参数 lambda, b
    7.  log_normal          对数正态          2 参数 mu, sigma
    8.  bose                玻色子模型        4 参数 d, beta, mu_tilde, scale
    9.  ehp                 指数超泊松(EHP)   3 参数 lambda, b, alpha
    10. generalized_gamma   广义伽马(Stacy)   3 参数 a, b, p
    11. sichel              西切尔(Sichel)    3 参数 theta, sigma2, gamma

约定：
    - ell 为词长（音节数），从 1 开始；离散模型内部平移 ell-1。
    - 除玻色子模型外，shape 参数拟合时 scale 固定为总计数；
      玻色子模型 scale 作为自由参数拟合（见 BOSE_FIT_SCALE）。
    - 所有 PMF 在词长支撑 1..PMF_SUPPORT_MAX 上数值归一化。

玻色子模型（核心）：
    词例分布（全数值积分）：
        P(ℓ) ∝ ℓ^{d/2} · B(d/2, β(ℓ-1) + μ̃)
    词型分布（低温近似）：
        P(ℓ) ∝ ℓ^{d/2} · exp(-βℓ + μ̃)
    其中 B(ν,z) 为广义玻色积分，见 :func:`bose_integral`。
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc, gammaln, kv  # kv=修正贝塞尔K
from scipy.stats import poisson, nbinom

# 跨目录导入：config.py 位于 ../01_语料处理/
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG01 = _os.path.join(_os.path.dirname(_HERE), "01_语料处理")
if _PKG01 not in _sys.path:
    _sys.path.insert(0, _PKG01)
from config import BOSE_MAX_N, PMF_SUPPORT_MAX  # noqa: E402


# 词长支撑（PMF 数值归一化用）
_SUPPORT = np.arange(1, PMF_SUPPORT_MAX + 1, dtype=float)


def _normalize(p_raw: np.ndarray) -> np.ndarray:
    """在支撑 _SUPPORT 上把未归一化的密度归一化为 PMF。"""
    s = np.sum(p_raw)
    if s <= 0 or not np.isfinite(s):
        # 退化为均匀分布，避免 NaN
        return np.ones_like(p_raw) / len(p_raw)
    return p_raw / s


# ================================================================
# 0. 玻色积分（核心数学内核，提取自 词长分布1.2.py）
# ================================================================
def bose_integral(nu, z, max_n: int = BOSE_MAX_N):
    """
    广义玻色积分 B(ν, z)，数值稳定性优化。

    定义：B(ν, z) = Γ(ν) · Σ_{n=1}^∞ e^{-nz} / n^ν
    （即 Γ(ν)·Li_ν(e^{-z})）

    小 z 近似：z→0 时 B(ν,z) ≈ -γ - ln z，避免级数求和发散。

    参数：
        nu:     阶数 ν = d/2
        z:      玻色逸度参数（标量或数组）
        max_n:  级数求和最大项数

    返回：
        B(ν, z)，与 z 同形状
    """
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    B = np.zeros_like(z)

    # z <= 0 时玻色积分发散（化学势非负），返回 inf
    mask_nonpos = z <= 0
    if np.any(mask_nonpos):
        B[mask_nonpos] = np.inf

    # 小 z 分支
    mask_small = (z > 0) & (z < 1e-5)
    if np.any(mask_small):
        B[mask_small] = -np.euler_gamma + np.log(1.0 / z[mask_small])

    # 大 z 分支：级数求和
    mask_large = ~mask_small
    if np.any(mask_large):
        n = np.arange(1, max_n + 1).reshape(-1, 1)
        z_large = z[mask_large].reshape(1, -1)
        exp_term = np.exp(-n * z_large)
        n_pow = n ** nu
        B[mask_large] = np.sum(exp_term / n_pow, axis=0) * gamma_func(nu)

    return B.item() if B.size == 1 else B


# ================================================================
# 1. 1-位移泊松 Shifted Poisson
# ================================================================
def pmf_shifted_poisson(ell, lam):
    """P(ℓ) = Pois(ℓ-1; λ)。词长 ℓ=1 对应计数 k=0。"""
    ell = np.asarray(ell, dtype=float)
    return poisson.pmf(ell - 1.0, lam)


# ================================================================
# 2. Gamma 模型（连续 Gamma，率参数 beta）
# ================================================================
def pmf_gamma(ell, alpha, beta):
    """P(ℓ) = β^α/Γ(α) · ℓ^{α-1} · e^{-βℓ}（在离散词长上归一化）。"""
    ell = np.asarray(ell, dtype=float)
    alpha = max(alpha, 1e-6)
    raw = np.zeros_like(ell, dtype=float)
    pos = ell > 0
    ellp = ell[pos]
    raw[pos] = (beta ** alpha) / gamma_func(alpha) * (ellp ** (alpha - 1)) * np.exp(-beta * ellp)
    # 归一化
    raw_support = (beta ** alpha) / gamma_func(alpha) * (_SUPPORT ** (alpha - 1)) * np.exp(-beta * _SUPPORT)
    return raw / np.sum(raw_support)


# ================================================================
# 3. 混合泊松 Poisson Mixture
# ================================================================
def pmf_poisson_mixture(ell, pi, lam1, lam2):
    """P(ℓ) = π·Pois(ℓ-1;λ1) + (1-π)·Pois(ℓ-1;λ2)。"""
    ell = np.asarray(ell, dtype=float)
    pi = np.clip(pi, 0.0, 1.0)
    return pi * poisson.pmf(ell - 1.0, lam1) + (1.0 - pi) * poisson.pmf(ell - 1.0, lam2)


# ================================================================
# 4. Zipf-Alekseev
# ================================================================
def pmf_zipf_alekseev(ell, a, b):
    """P(ℓ) ∝ (ℓ + a)^{-b}，在离散支撑上归一化。"""
    ell = np.asarray(ell, dtype=float)
    raw = (ell + a) ** (-b)
    raw_support = (_SUPPORT + a) ** (-b)
    return raw / np.sum(raw_support)


# ================================================================
# 5. 负二项 Negative Binomial
# ================================================================
def pmf_negative_binomial(ell, r, p):
    """P(ℓ) = NB(k=ℓ-1; r, p)。"""
    ell = np.asarray(ell, dtype=float)
    return nbinom.pmf(ell - 1.0, r, p)


# ================================================================
# 6. Hyper-Poisson（含 EHP 所需的 CDF）
# ================================================================
def hyper_poisson_raw(k, lam, b):
    """
    Hyper-Poisson 未归一化质量：g(k) = λ^k / Γ(k + b)，k = 0,1,2,...
    """
    k = np.asarray(k, dtype=float)
    return np.exp(k * np.log(lam) - gammaln(k + b))


def pmf_hyper_poisson(ell, lam, b):
    """Hyper-Poisson PMF，在 k=0..K 支撑上归一化（ell-1 平移）。"""
    ell = np.asarray(ell, dtype=float)
    K = PMF_SUPPORT_MAX
    k_support = np.arange(0, K + 1, dtype=float)
    g_support = hyper_poisson_raw(k_support, lam, b)
    Z = np.sum(g_support)
    return hyper_poisson_raw(ell - 1.0, lam, b) / Z


def _hyper_poisson_cdf(lam, b, k_max=PMF_SUPPORT_MAX):
    """Hyper-Poisson 的完整 CDF 数组 F(k), k=0..k_max。"""
    k = np.arange(0, k_max + 1, dtype=float)
    g = hyper_poisson_raw(k, lam, b)
    Z = np.sum(g)
    pmf = g / Z
    cdf = np.cumsum(pmf)
    return cdf


# ================================================================
# 7. 对数正态 Log-Normal
# ================================================================
def pmf_log_normal(ell, mu, sigma):
    """P(ℓ) = 1/(ℓσ√2π) · exp(-(ln ℓ - μ)^2/(2σ^2))，离散归一化。"""
    ell = np.asarray(ell, dtype=float)
    sigma = max(sigma, 1e-6)

    def pdf(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos = x > 0
        xp = x[pos]
        out[pos] = (1.0 / (xp * sigma * np.sqrt(2 * np.pi))) * \
                   np.exp(-(np.log(xp) - mu) ** 2 / (2 * sigma ** 2))
        return out

    raw = pdf(ell)
    return raw / np.sum(pdf(_SUPPORT))


# ================================================================
# 8. 玻色子模型 Bose Model（核心；词例/词型两种形式）
# ================================================================
def bose_pmf_token(ell, d, beta, mu_tilde):
    """
    词例分布形状（未乘 scale）：
        f(ℓ) = ℓ^{d/2} · B(d/2, β(ℓ-1) + μ̃)
    全数值积分形式。
    """
    ell = np.asarray(ell, dtype=float)
    nu = d / 2.0
    z = beta * (ell - 1.0) + mu_tilde
    B = bose_integral(nu, z)
    return (ell ** nu) * B


def bose_pmf_type(ell, d, beta, mu_tilde):
    """
    词型分布形状（低温近似，未乘 scale）：
        f(ℓ) = ℓ^{d/2} · exp(-βℓ + μ̃)
    """
    ell = np.asarray(ell, dtype=float)
    nu = d / 2.0
    return (ell ** nu) * np.exp(-beta * ell + mu_tilde)


def bose_model(ell, d, beta, mu_tilde, scale, dist_type="token"):
    """
    玻色子模型预测计数 = scale · 形状。

    参数：
        ell:       词长数组
        d:         语义维度
        beta:      无量纲能量比 b/T
        mu_tilde:  无量纲化学势 δ/T
        scale:     尺度参数（拟合总计数级）
        dist_type: "token"（全数值积分）或 "type"（低温近似）
    """
    if dist_type == "token":
        return scale * bose_pmf_token(ell, d, beta, mu_tilde)
    else:
        return scale * bose_pmf_type(ell, d, beta, mu_tilde)


# ================================================================
# 9. 指数超泊松 EHP
# ================================================================
def pmf_ehp(ell, lam, b, alpha):
    """
    EHP: P(ℓ) = F_HP(ℓ-1)^α - F_HP(ℓ-2)^α
    其中 F_HP 为 Hyper-Poisson 累积分布。
    ℓ=1 时 F_HP(-1)=0。
    """
    ell = np.asarray(ell, dtype=float)
    cdf = _hyper_poisson_cdf(lam, b)  # index k=0..K
    K = len(cdf) - 1

    def F(k):
        # F_HP(k)：k<0 返回 0，k>K 返回 1
        k = np.asarray(k, dtype=float)
        out = np.where(k < 0, 0.0, np.where(k >= K, 1.0, cdf[np.clip(k.astype(int), 0, K)]))
        return out

    pmf = F(ell - 1.0) ** alpha - F(ell - 2.0) ** alpha
    pmf = np.clip(pmf, 0.0, None)
    # 归一化
    ks = np.arange(1, PMF_SUPPORT_MAX + 1, dtype=float)
    norm = F(ks - 1.0) ** alpha - F(ks - 2.0) ** alpha
    norm = np.clip(norm, 0.0, None)
    s = np.sum(norm)
    return pmf / s if s > 0 else pmf


# ================================================================
# 10. 广义伽马 Generalized Gamma (Stacy)
# ================================================================
def pmf_generalized_gamma(ell, a, b, p):
    """
    Stacy 广义伽马 PDF：
        f(ℓ) = p/(b^a Γ(a/p)) · ℓ^{a-1} · exp(-(ℓ/b)^p)
    在离散词长上归一化。
    """
    ell = np.asarray(ell, dtype=float)
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    p = max(p, 1e-6)

    def pdf(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos = x > 0
        xp = x[pos]
        out[pos] = (p / (b ** a * gamma_func(a / p))) * \
                   (xp ** (a - 1)) * np.exp(-(xp / b) ** p)
        return out

    raw = pdf(ell)
    return raw / np.sum(pdf(_SUPPORT))


# ================================================================
# 11. 西切尔 Sichel
# ================================================================
def pmf_sichel(ell, theta, sigma2, gamma):
    """
    Sichel（广义逆高斯-泊松混合）PMF。

    采用数值稳定的递推/闭式形式：
        b = sqrt(1 + 2 σ² θ)
        P(k) = (θ^k/k!) · ((b-1)/(b+1))^{k/2} · K_{k+γ}(θb) / K_γ(θb)
    其中 k = ell-1，K_ν 为修正贝塞尔函数。在离散支撑上归一化。
    """
    ell = np.asarray(ell, dtype=float)
    sigma2 = max(sigma2, 1e-6)
    b = np.sqrt(1.0 + 2.0 * sigma2 * theta)

    def pmf_k(k_arr):
        k_arr = np.asarray(k_arr, dtype=float)
        log_terms = k_arr * np.log(theta) - gammaln(k_arr + 1) \
                    + 0.5 * k_arr * np.log((b - 1.0) / (b + 1.0))
        # 贝塞尔比用 log 域计算以保证数值稳定
        log_ratio = kv(k_arr + gamma, theta * b) / kv(gamma, theta * b)
        log_ratio = np.log(np.maximum(log_ratio, 1e-300))
        return np.exp(log_terms + log_ratio)

    pmf = pmf_k(ell - 1.0)
    pmf = np.where(np.isfinite(pmf), pmf, 0.0)
    # 归一化
    support_k = np.arange(0, PMF_SUPPORT_MAX + 1, dtype=float)
    norm = pmf_k(support_k)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    s = np.sum(norm)
    return pmf / s if s > 0 else pmf


# ================================================================
# 模型注册表：统一接口
# ================================================================
# 每个模型：shape 函数、shape 参数名、是否把 scale 作为自由参数
# 返回预测计数时：scale * pmf(ell)。
#   - fit_scale=False：scale 固定为总计数（n_params = shape 参数数）
#   - fit_scale=True ：scale 自由拟合（n_params 含 scale，如玻色子模型）
MODEL_REGISTRY = {
    "shifted_poisson": {
        "pmf": pmf_shifted_poisson,
        "shape_params": ["lambda"],
        "fit_scale": False,
    },
    "gamma": {
        "pmf": pmf_gamma,
        "shape_params": ["alpha", "beta"],
        "fit_scale": False,
    },
    "poisson_mixture": {
        "pmf": pmf_poisson_mixture,
        "shape_params": ["pi", "lambda1", "lambda2"],
        "fit_scale": False,
    },
    "zipf_alekseev": {
        "pmf": pmf_zipf_alekseev,
        "shape_params": ["a", "b"],
        "fit_scale": False,
    },
    "negative_binomial": {
        "pmf": pmf_negative_binomial,
        "shape_params": ["r", "p"],
        "fit_scale": False,
    },
    "hyper_poisson": {
        "pmf": pmf_hyper_poisson,
        "shape_params": ["lambda", "b"],
        "fit_scale": False,
    },
    "log_normal": {
        "pmf": pmf_log_normal,
        "shape_params": ["mu", "sigma"],
        "fit_scale": False,
    },
    "bose": {
        "pmf": None,  # 特殊：见 bose_model
        "shape_params": ["d", "beta", "mu_tilde", "scale"],
        "fit_scale": True,
    },
    "ehp": {
        "pmf": pmf_ehp,
        "shape_params": ["lambda", "b", "alpha"],
        "fit_scale": False,
    },
    "generalized_gamma": {
        "pmf": pmf_generalized_gamma,
        "shape_params": ["a", "b", "p"],
        "fit_scale": False,
    },
    "sichel": {
        "pmf": pmf_sichel,
        "shape_params": ["theta", "sigma2", "gamma"],
        "fit_scale": False,
    },
}


def predict(model_id, ell, params, total_count=None, dist_type="token"):
    """
    统一预测接口：给定模型 id、词长、参数，返回预测计数。

    参数：
        model_id:    模型标识（见 MODEL_REGISTRY）
        ell:         词长数组
        params:      dict 或 list；list 时按 shape_params 顺序
        total_count: 总计数；fit_scale=False 时用作 scale
        dist_type:   分布类型 token/type（仅玻色子模型使用）

    返回：
        np.ndarray: 预测计数
    """
    spec = MODEL_REGISTRY[model_id]
    if isinstance(params, dict):
        ordered = [params[n] for n in spec["shape_params"]]
    else:
        ordered = list(params)

    if model_id == "bose":
        d, beta, mu_tilde, scale = ordered
        return bose_model(ell, d, beta, mu_tilde, scale, dist_type=dist_type)

    if spec["fit_scale"]:
        *shape, scale = ordered
        return scale * spec["pmf"](ell, *shape)
    else:
        scale = total_count if total_count is not None else 1.0
        return scale * spec["pmf"](ell, *ordered)


if __name__ == "__main__":
    # 自检：打印各模型在样例参数下的归一化性
    ell_demo = np.arange(1, 11, dtype=float)
    demo_params = {
        "shifted_poisson": [1.5],
        "gamma": [2.0, 1.0],
        "poisson_mixture": [0.7, 0.5, 2.5],
        "zipf_alekseev": [0.0, 2.0],
        "negative_binomial": [2.0, 0.5],
        "hyper_poisson": [1.5, 1.0],
        "log_normal": [0.5, 0.8],
        "ehp": [1.5, 1.0, 0.5],
        "generalized_gamma": [2.0, 3.0, 1.0],
        "sichel": [1.0, 1.0, -2.0],
    }
    for mid, p in demo_params.items():
        pmf = MODEL_REGISTRY[mid]["pmf"](ell_demo, *p)
        print(f"{mid:20s} sum_pmf(1..10)={np.sum(pmf):.4f}  "
              f"first={pmf[0]:.4f}")
    # 玻色子模型
    tok = bose_model(ell_demo, 3.5, 1.4, 1.4, 1e5, "token")
    typ = bose_model(ell_demo, 7.0, 1.6, 0.7, 3e4, "type")
    print(f"{'bose(token)':20s} pred={np.round(tok, 1)}")
    print(f"{'bose(type)':20s} pred={np.round(typ, 1)}")
