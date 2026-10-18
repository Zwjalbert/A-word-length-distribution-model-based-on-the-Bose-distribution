# -*- coding: utf-8 -*-
"""
================================================================
plotter_sci_cn.py —— SCI 风格中文绘图
================================================================

面向国内期刊 / 会议海报的中文 SCI 风格绘图函数。
版式规范与 plotter_sci.py 一致（线宽、字号、无多余装饰），
但坐标轴标题使用中文（配合 SimHei / 微软雅黑），
适合 CCL 等中文会议的海报与汇报材料。
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


def apply_sci_cn_style():
    """应用 SCI 中文风格（中文黑体 + 紧凑版式）。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial",
                            "DejaVu Sans"],
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


def plot_fit_sci_cn(ell, y_obs, y_pred, y_err=None,
                    xlabel=r"词长 $\ell$（音节数）",
                    ylabel=r"计数 $N(\ell)$",
                    title="", save_path=None, log_y=False):
    """SCI 风格拟合对比图（中文标签）。"""
    apply_sci_cn_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    if y_err is not None:
        ax.errorbar(ell, y_obs, yerr=y_err, fmt="o", color="#1f77b4",
                    markersize=4, capsize=2, lw=1, label="实测值")
    else:
        ax.plot(ell, y_obs, "o", color="#1f77b4", markersize=4,
                label="实测值")
    ax.plot(ell, y_pred, "-", color="#d62728", lw=1.5, label="拟合曲线")
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


def plot_two_panel_cn(ell, token_obs, token_pred,
                      type_obs, type_pred, save_path=None):
    """
    双语料双分布面板图：左=词例分布，右=词型分布（中文）。
    对应原始实验的「双分布单独拟合总图」版式。
    """
    apply_sci_cn_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax1.errorbar(ell, token_obs, yerr=np.sqrt(token_obs), fmt="o",
                 color="#ff7f0e", markersize=4, capsize=2, label="实测")
    ax1.plot(ell, token_pred, "-", color="#1f77b4", lw=1.5, label="拟合")
    ax1.set_xlabel(r"词长 $\ell$"); ax1.set_ylabel("词例数")
    ax1.set_title("(a) 词长-词例分布", fontweight="bold")
    ax1.legend()

    ax2.errorbar(ell, type_obs, yerr=np.sqrt(type_obs), fmt="o",
                 color="#2ca02c", markersize=4, capsize=2, label="实测")
    ax2.plot(ell, type_pred, "-", color="#9467bd", lw=1.5, label="拟合")
    ax2.set_xlabel(r"词长 $\ell$"); ax2.set_ylabel("词型数")
    ax2.set_title("(b) 词长-词型分布", fontweight="bold")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_aic_rank_cn(model_names, aic_values, save_path=None):
    """中文 SCI 风格 AIC 排名图。"""
    apply_sci_cn_style()
    order = np.argsort(aic_values)
    names_sorted = [model_names[i] for i in order]
    aic_sorted = np.array(aic_values)[order]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.barh(names_sorted, aic_sorted, color="#9467bd")
    ax.set_xlabel("AIC（越低越优）")
    ax.set_title("模型排名（按 AIC）")
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    ell_demo = np.arange(1, 9, dtype=float)
    tok = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    typ = np.array([3545, 6476, 4118, 2041, 693, 150, 16, 5], dtype=float)
    plot_two_panel_cn(ell_demo, tok, tok * 0.98, typ, typ * 0.98,
                      save_path="sci_cn_demo.png")
    print("SCI 中文 demo 图已保存")
