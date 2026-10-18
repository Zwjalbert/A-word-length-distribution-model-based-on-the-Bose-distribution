# -*- coding: utf-8 -*-
"""
================================================================
plotter.py —— 基础绘图模块（拟合对比与残差图）
================================================================

提供实验内部使用的基础绘图函数：
    - 单模型拟合对比图（观测散点 + 拟合曲线 + 误差棒）
    - 标准化残差图（带 ±2σ 参考线）
    - 多模型拟合叠加对比图
    - 模型指标对比柱状图（R²_adj / AIC）

本模块使用中文标签（配合 config.PLT_RCPARAMS 的 SimHei）。
SCI 投稿用的英文/中文版式见 plotter_sci.py / plotter_sci_cn.py。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 跨目录导入
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in [_os.path.join(_ROOT, "01_语料处理")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from config import apply_plot_style, COLORS, MODEL_COLORS, ensure_dir  # noqa: E402


def plot_fit_comparison(ell, y_obs, y_pred, y_err=None,
                        title="拟合对比", xlabel="词长 ℓ",
                        ylabel="计数", save_path=None,
                        obs_color=None, fit_color=None):
    """
    绘制单模型拟合对比图。

    参数：
        ell, y_obs, y_pred, y_err: 词长、观测、预测、误差
        title/xlabel/ylabel: 标题与坐标轴
        save_path: 保存路径（None 则仅 plt.show）
    """
    apply_plot_style()
    obs_color = obs_color or COLORS["token"]["obs"]
    fit_color = fit_color or COLORS["token"]["fit"]

    fig, ax = plt.subplots(figsize=(7, 5))
    if y_err is not None:
        ax.errorbar(ell, y_obs, yerr=y_err, fmt="o", markersize=7,
                    capsize=3, color=obs_color, label="实测值 ±1σ", zorder=5)
    else:
        ax.plot(ell, y_obs, "o", color=obs_color, markersize=7,
                label="实测值", zorder=5)
    ax.plot(ell, y_pred, "-", lw=2.5, color=fit_color,
            label="拟合曲线", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_residuals(ell, residuals, title="标准化残差", save_path=None):
    """
    绘制标准化（皮尔逊）残差散点图，带 ±2σ 参考线。

    参数：
        ell:        词长数组
        residuals:  标准化残差数组
        title:      标题
        save_path:  保存路径
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ell, residuals, s=70, color=COLORS["residual"],
               edgecolor="white", zorder=5)
    ax.axhline(0, color="black", lw=1.5)
    ax.axhline(2, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(-2, color="gray", ls="--", lw=1, alpha=0.7)
    n_out = int(np.sum(np.abs(residuals) > 2))
    ax.text(0.95, 0.95, f"|残差|>2σ: {n_out}/{len(residuals)}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("词长 ℓ")
    ax.set_ylabel("标准化残差 (O-E)/σ")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multi_model_fit(ell, y_obs, model_curves, y_err=None,
                         title="多模型拟合对比", save_path=None):
    """
    在同一坐标轴上叠加多个模型的拟合曲线。

    参数：
        ell:         词长数组
        y_obs:       观测计数
        model_curves: list[(model_name, y_pred)] 待叠加的曲线
        y_err:       观测误差棒
        save_path:   保存路径
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if y_err is not None:
        ax.errorbar(ell, y_obs, yerr=y_err, fmt="o", color="black",
                    markersize=6, capsize=3, label="实测值", zorder=10)
    else:
        ax.plot(ell, y_obs, "ko", markersize=6, label="实测值", zorder=10)
    for i, (name, yp) in enumerate(model_curves):
        ax.plot(ell, yp, "-", lw=2,
                color=MODEL_COLORS[i % len(MODEL_COLORS)], label=name)
    ax.set_xlabel("词长 ℓ")
    ax.set_ylabel("计数")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metrics_bar(model_names, metric_values, metric_label="R²_adj",
                     title="模型指标对比", save_path=None):
    """
    绘制各模型某指标（如 R²_adj / AIC）的柱状图。
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    bars = ax.bar(x, metric_values, color=MODEL_COLORS[:len(model_names)])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, metric_values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    # 自检
    ell_demo = np.arange(1, 9, dtype=float)
    obs = np.array([164256, 51665, 22092, 9228, 2402, 327, 30, 7], dtype=float)
    pred = np.array([160000, 52000, 22000, 9000, 2500, 330, 30, 7], dtype=float)
    plot_fit_comparison(ell_demo, obs, pred, save_path="demo_fit.png")
    plot_residuals(ell_demo, (obs - pred) / np.sqrt(obs),
                   save_path="demo_res.png")
    print("demo 图已保存")
