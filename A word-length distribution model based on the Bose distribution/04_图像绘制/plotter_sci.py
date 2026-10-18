# -*- coding: utf-8 -*-
"""
================================================================
plotter_sci.py —— SCI 期刊风格英文绘图
================================================================

面向 SCI 论文投稿的英文版式绘图函数。与 plotter.py 的区别：
    - 全部标签使用英文（避免字体缺失）
    - 字体切换为 Arial / DejaVu Sans
    - 字号、线宽按单栏/双栏排版习惯调整
    - 图例更紧凑，无网格（或极浅网格）

输出图表可直接插入 SCI 论文（300 dpi PNG/PDF）。
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 跨目录导入
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in [_os.path.join(_ROOT, "01_语料处理")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from config import ensure_dir  # noqa: E402


def apply_sci_style():
    """应用 SCI 英文绘图风格（Arial, 紧凑版式）。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def plot_fit_sci(ell, y_obs, y_pred, y_err=None,
                 xlabel=r"Word length $\ell$ (syllables)",
                 ylabel=r"Count $N(\ell)$",
                 title="", save_path=None, log_y=False):
    """
    SCI 风格拟合对比图（英文标签）。

    参数：
        log_y: 是否纵轴对数坐标（重尾分布常用）
    """
    apply_sci_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # 单栏宽度
    if y_err is not None:
        ax.errorbar(ell, y_obs, yerr=y_err, fmt="o", color="#1f77b4",
                    markersize=4, capsize=2, lw=1, label="Observed")
    else:
        ax.plot(ell, y_obs, "o", color="#1f77b4", markersize=4,
                label="Observed")
    ax.plot(ell, y_pred, "-", color="#d62728", lw=1.5, label="Fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multi_fit_sci(ell, y_obs, model_curves,
                       xlabel=r"Word length $\ell$",
                       ylabel=r"Count $N(\ell)$",
                       save_path=None, log_y=True):
    """SCI 风格多模型叠加对比（英文）。"""
    apply_sci_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(ell, y_obs, "ko", markersize=4, label="Observed", zorder=10)
    cmap = plt.cm.turbo(np.linspace(0, 0.9, max(len(model_curves), 1)))
    for (name, yp), c in zip(model_curves, cmap):
        ax.plot(ell, yp, "-", lw=1.3, color=c, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=7, ncol=1)
    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_aic_rank_sci(model_names, aic_values, save_path=None):
    """SCI 风格 AIC 排名图（越低越好）。"""
    apply_sci_style()
    order = np.argsort(aic_values)
    names_sorted = [model_names[i] for i in order]
    aic_sorted = np.array(aic_values)[order]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.barh(names_sorted, aic_sorted, color="#1f77b4")
    ax.set_xlabel("AIC")
    ax.set_title("Model ranking (by AIC)")
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    ell_demo = np.arange(1, 9, dtype=float)
    obs = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    pred = np.array([160000, 52000, 22000, 9000, 2500, 330, 30, 7], dtype=float)
    plot_fit_sci(ell_demo, obs, pred, save_path="sci_demo.png")
    print("SCI demo 图已保存")
