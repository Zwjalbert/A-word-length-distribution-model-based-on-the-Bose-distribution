# -*- coding: utf-8 -*-
"""
================================================================
plot_paper_figures.py —— 论文《基于玻色—爱因斯坦统计的词长分布统一模型》
（paper_0295.pdf）全部 5 张插图（图1–图5）的绘制脚本（SCI 风格）
================================================================

依赖数据（由 05_一键运行/run_all.py 生成，位于 experiment_code/results/）：
  results/<corpus>/bnc_<corpus>_<dist>_results.json     完整拟合结果（含观测/预测/参数）
  results/<corpus>/bnc_<corpus>_<dist>_comparison.csv   11 模型对比表（R²_adj、AIC 等）
  其中 <corpus> ∈ {academic, news, spoken, fiction}，<dist> ∈ {token, type}

论文插图与本脚本输出的对应关系：
  图1  11 个模型在四语料词例分布上的调整 R² 对比   -> fig1_r2_comparison.png
  图2  四语料词例分布与玻色子模型拟合曲线(2×2)      -> fig2_token_fit.png
  图3  四语料词型分布与玻色子模型拟合曲线(2×2)      -> fig3_type_fit.png
  图4  玻色子模型与 Top-4 经验模型拟合对比
       （左:口语词例; 右:新闻词型）                  -> fig4_top4_comparison.png
  图5  模型参数在四语料词例/词型上的对比
       （左:语义维度 d; 右:形式成本比 β）            -> fig5_param_comparison.png

用法：
  1) 先生成结果数据（若 results/ 尚不存在）：
       cd 05_一键运行 && python run_all.py
  2) 再绘制论文插图：
       python 04_图像绘制/plot_paper_figures.py
     （或由 run_all.py 末尾自动调用）

依赖：numpy、pandas、matplotlib（300 dpi，PNG 输出）
================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------
# SCI 风格设置
# ---------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.3
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)          # experiment_code/
RESULT_ROOT = os.path.join(_PROJECT_ROOT, "results")

CORPUS_ORDER = ['academic', 'news', 'spoken', 'fiction']
CORPUS_LABELS = {
    'academic': 'Academic',
    'news': 'News',
    'spoken': 'Spoken',
    'fiction': 'Fiction',
}
COLORS = {
    'academic': '#1f77b4',
    'news': '#ff7f0e',
    'spoken': '#2ca02c',
    'fiction': '#d62728',
}

# 中文模型名 -> 论文/图例中的英文名
MODEL_NAME_EN = {
    '混合泊松': 'Mixture Poisson',
    '指数超泊松(EHP)': 'EHP',
    '玻色子模型': 'Bose model (ours)',
    'Hyper-Poisson': 'Hyper-Poisson',
    '广义伽马(GG)': 'GG',
    'Gamma': 'Gamma',
    '1-位移泊松': '1-shift Poisson',
    '负二项': 'NegBin',
    '对数正态': 'Lognormal',
    'Zipf-Alekseev': 'Zipf-Alekseev',
    '西切尔(Sichel)': 'Sichel',
}


def load_results(corpus_key, dist_type):
    """加载某语料某分布的结果 JSON（由 run_all.py 生成）。"""
    path = os.path.join(RESULT_ROOT, corpus_key,
                        f"bnc_{corpus_key}_{dist_type}_results.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_comparison(corpus_key, dist_type):
    """加载某语料某分布的 11 模型对比 CSV。"""
    path = os.path.join(RESULT_ROOT, corpus_key,
                        f"bnc_{corpus_key}_{dist_type}_comparison.csv")
    return pd.read_csv(path)


# =================================================================
# 图1（论文图1）：11 个模型在四语料词例分布上的调整 R² 对比
# =================================================================
def draw_fig1():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_names_cn = None
    r2_data = []
    for c in CORPUS_ORDER:
        comparison = load_comparison(c, 'token')
        if model_names_cn is None:
            model_names_cn = comparison.sort_values(
                'R2_adj', ascending=False)['model_name'].tolist()
        r2_data.append(
            comparison.set_index('model_name')
            .reindex(model_names_cn)['R2_adj'].values)

    r2_array = np.array(r2_data, dtype=float)
    x = np.arange(len(model_names_cn))
    width = 0.2
    for i, c in enumerate(CORPUS_ORDER):
        ax.bar(x + (i - 1.5) * width, r2_array[i], width,
               label=CORPUS_LABELS[c], color=list(COLORS.values())[i],
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAME_EN.get(n, n) for n in model_names_cn],
                       rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Adjusted $R^2$')
    ax.set_title('Adjusted $R^2$ across 11 models (token distribution)')
    ax.legend(frameon=False, ncol=4, loc='lower right')
    ax.grid(True, axis='y')
    ax.set_ylim(0.8, 1.005)
    plt.tight_layout()
    out_path = os.path.join(RESULT_ROOT, 'fig1_r2_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[论文图1] 已保存: {out_path}")


# =================================================================
# 图2（论文图2）：四语料词例分布与玻色子模型拟合曲线（2×2）
# =================================================================
def draw_fig2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for idx, corpus in enumerate(CORPUS_ORDER):
        ax = axes[idx]
        data = load_results(corpus, 'token')
        bose = data['results']['bose']
        ell = np.array(bose['lengths'])
        obs = np.array(bose['observed'])
        pred = np.array(bose['predicted'])
        color = COLORS[corpus]
        ax.plot(ell, obs, 'o', color='black', markersize=4,
                label='Observed', zorder=3)
        ax.plot(ell, pred, '-', color=color, linewidth=2,
                label='Bose model', zorder=2)
        ax.fill_between(ell, obs - np.sqrt(obs), obs + np.sqrt(obs),
                        alpha=0.15, color=color)
        ax.set_xlabel('Word length $\\ell$ (syllables)')
        ax.set_ylabel('Count')
        ax.set_title(f'({chr(97 + idx)}) {CORPUS_LABELS[corpus]} token')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True)
        ax.set_xlim(left=0.5)
    plt.tight_layout()
    out_path = os.path.join(RESULT_ROOT, 'fig2_token_fit.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[论文图2] 已保存: {out_path}")


# =================================================================
# 图3（论文图3）：四语料词型分布与玻色子模型拟合曲线（2×2）
# =================================================================
def draw_fig3():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for idx, corpus in enumerate(CORPUS_ORDER):
        ax = axes[idx]
        data = load_results(corpus, 'type')
        bose = data['results']['bose']
        ell = np.array(bose['lengths'])
        obs = np.array(bose['observed'])
        pred = np.array(bose['predicted'])
        color = COLORS[corpus]
        ax.plot(ell, obs, 'o', color='black', markersize=4,
                label='Observed', zorder=3)
        ax.plot(ell, pred, '-', color=color, linewidth=2,
                label='Bose model', zorder=2)
        ax.set_xlabel('Word length $\\ell$ (syllables)')
        ax.set_ylabel('Count')
        ax.set_title(f'({chr(97 + idx)}) {CORPUS_LABELS[corpus]} type')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True)
        ax.set_xlim(left=0.5)
    plt.tight_layout()
    out_path = os.path.join(RESULT_ROOT, 'fig3_type_fit.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[论文图3] 已保存: {out_path}")


# =================================================================
# 图4（论文图4）：玻色子模型与 Top-4 经验模型的拟合对比
#     左: 口语词例（单调递减极端形态）; 右: 新闻词型（单峰形态）
# =================================================================
def _draw_top4_panel(ax, corpus, dist_type):
    data = load_results(corpus, dist_type)
    comparison = load_comparison(corpus, dist_type)
    top = comparison.sort_values('AIC').head(4)
    bose_row = comparison[comparison['model_id'] == 'bose']
    if len(bose_row) > 0 and 'bose' not in top['model_id'].values:
        top = pd.concat([top, bose_row], ignore_index=True)

    ell = np.array(data['results']['bose']['lengths'])
    obs = np.array(data['results']['bose']['observed'])
    ax.plot(ell, obs, 'ko', markersize=5, label='Observed', zorder=5)

    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (_, row) in enumerate(top.iterrows()):
        model_id = row['model_id']
        pred = np.array(data['results'][model_id]['predicted'])
        ls = '--' if model_id == 'bose' else '-'
        lw = 2.5 if model_id == 'bose' else 1.5
        label = MODEL_NAME_EN.get(row['model_name'], row['model_name'])
        if model_id == 'bose':
            label = 'Bose model (ours)'
        ax.plot(ell, pred, ls, color=model_colors[i % len(model_colors)],
                linewidth=lw, label=label, zorder=4 - i)
    ax.set_xlabel('Word length $\\ell$ (syllables)')
    ax.set_ylabel('Count')
    ax.grid(True)
    ax.set_xlim(left=0.5)


def draw_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_top4_panel(axes[0], 'spoken', 'token')
    axes[0].set_title('(a) Spoken token distribution')
    axes[0].legend(loc='upper right', frameon=False, fontsize=8)
    _draw_top4_panel(axes[1], 'news', 'type')
    axes[1].set_title('(b) News type distribution')
    axes[1].legend(loc='upper right', frameon=False, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULT_ROOT, 'fig4_top4_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[论文图4] 已保存: {out_path}")


# =================================================================
# 图5（论文图5）：玻色子模型参数在四语料词例/词型上的对比
#     左: 语义维度 d; 右: 形式成本比 β = b/T
# =================================================================
def draw_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    corpora = CORPUS_ORDER
    x = np.arange(len(corpora))
    width = 0.35

    # 左：语义维度 d
    d_token, d_type = [], []
    for c in corpora:
        d_token.append(load_results(c, 'token')['results']['bose']
                       ['param_dict']['d'])
        d_type.append(load_results(c, 'type')['results']['bose']
                      ['param_dict']['d'])
    ax = axes[0]
    ax.bar(x - width / 2, d_token, width, label='Token',
           color='#4c72b0', edgecolor='black', linewidth=0.8)
    ax.bar(x + width / 2, d_type, width, label='Type',
           color='#dd8452', edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.set_ylabel('Semantic dimension $d$')
    ax.set_title('(a) Semantic dimension $d$')
    ax.legend(frameon=False)
    ax.grid(True, axis='y')

    # 右：形式成本比 β
    b_token, b_type = [], []
    for c in corpora:
        b_token.append(load_results(c, 'token')['results']['bose']
                       ['param_dict']['beta'])
        b_type.append(load_results(c, 'type')['results']['bose']
                      ['param_dict']['beta'])
    ax = axes[1]
    ax.bar(x - width / 2, b_token, width, label='Token',
           color='#4c72b0', edgecolor='black', linewidth=0.8)
    ax.bar(x + width / 2, b_type, width, label='Type',
           color='#dd8452', edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.set_ylabel(r'Form cost ratio $\beta = b/T$')
    ax.set_title(r'(b) Form cost ratio $\beta$')
    ax.legend(frameon=False)
    ax.grid(True, axis='y')

    plt.tight_layout()
    out_path = os.path.join(RESULT_ROOT, 'fig5_param_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[论文图5] 已保存: {out_path}")


# =================================================================
# 主流程：按论文图号顺序绘制图1–图5
# 说明：绘图语句置于模块级，以保证与 05_一键运行/run_all.py 的原有
# 调用约定一致（`from plot_paper_figures import *` 即触发全部绘图）；
# 直接执行本文件（python plot_paper_figures.py）同样生效。
# =================================================================
os.makedirs(RESULT_ROOT, exist_ok=True)
draw_fig1()
draw_fig2()
draw_fig3()
draw_fig4()
draw_fig5()
print("\n论文 5 张插图全部绘制完成！")
