# -*- coding: utf-8 -*-
"""
================================================================
config.py —— 全局集中配置模块
================================================================

本模块集中管理词长分布量子模型实验的全部全局配置，包括：
    1. 路径配置（语料根目录、输出目录、CMU 词典位置）
    2. BNC Baby 四语料的目录映射
    3. 科研绘图 rcParams 风格
    4. 拟合超参数（最大迭代、多初始值、参数边界等）
    5. 统一配色方案

所有其他模块通过 `from config import ...` 或 `import config as CFG`
引用此处的常量，避免在代码中硬编码路径与魔法数字。

注意：本模块仅定义常量与工具函数，不执行任何耗时操作。
"""

import os
import matplotlib.pyplot as plt

# ================================================================
# 1. 路径配置
# ================================================================
# 实验代码根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（上一级：实验代码/）
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# CMU 发音词典（随代码仓库分发，位于 data/ 目录）
CMUDICT_PATH = os.path.join(PROJECT_ROOT, "data", "cmudict.dict")

# BNC Baby 原始 XML 语料根目录（只读参考数据，不随仓库分发）
# 实际目录结构：<BNC_ROOT>/Texts/{aca,dem,news,fic}/*.xml
BNC_RAW_ROOT = r"F:\科研\学术会议\2026 参会\CCL2026\实验0914\英文原始语料\download"

# 处理后语料（词表 CSV）输出目录
CORPUS_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 实验结果（CSV / JSON / 图表）输出根目录
RESULT_ROOT = os.path.join(PROJECT_ROOT, "results")

# ================================================================
# 2. BNC Baby 四语料映射
# ================================================================
# 逻辑语料名 -> BNC 原始 XML 子目录名
#   academic -> aca  学术书面语
#   news     -> news 新闻书面语
#   spoken   -> dem  口语对话（人口统计采样）
#   fiction  -> fic  小说文学
CORPUS_MAP = {
    "academic": "aca",
    "news": "news",
    "spoken": "dem",
    "fiction": "fic",
}

# 中文展示名
CORPUS_CN = {
    "academic": "学术书面语",
    "news": "新闻书面语",
    "spoken": "口语对话",
    "fiction": "小说文学",
}

# 分布类型
DIST_TYPES = ["token", "type"]  # token=词例分布, type=词型分布

# ================================================================
# 3. 绘图 rcParams 科研风格
# ================================================================
FIG_DPI = 300
SAVEFIG_FORMAT = ["png"]

PLT_RCPARAMS = {
    "font.family": "sans-serif",
    # 中文绘图使用 SimHei；SCI 英文绘图切换为 Arial/DejaVu Sans
    "font.sans-serif": ["SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
}

# 统一配色方案（词例=橙/蓝，词型=绿/紫，残差=灰）
COLORS = {
    "token": {"obs": "#ff7f0e", "fit": "#1f77b4"},
    "type": {"obs": "#2ca02c", "fit": "#9467bd"},
    "residual": "#7f7f7f",
}

# 11 个模型的可视化配色（循环使用）
MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#000000",
]


def apply_plot_style():
    """应用全局科研绘图风格。在所有绘图脚本入口处调用一次。"""
    plt.rcParams.update(PLT_RCPARAMS)


def ensure_dir(path):
    """确保目录存在，不存在则递归创建。返回 path 本身以便链式调用。"""
    os.makedirs(path, exist_ok=True)
    return path


# ================================================================
# 4. 拟合超参数
# ================================================================
FIT_MAXFEV = 200000          # curve_fit 最大迭代次数
FIT_FTOL = 1e-10             # 函数收敛容差
ENABLE_MULTI_INIT = True      # 是否启用多初始值重试
MULTI_INIT_NUM = 2           # 随机/网格初始值数量
RANDOM_SEED = 42             # 随机种子，保证可复现

# 卡方检验合并 bin 时的最小期望频数
CHI2_MIN_EXPECTED = 5
ENABLE_YATES_CORRECTION = True  # 两 bin 时启用 Yates 连续性修正

# 玻色积分级数求和上限（max_n），保证数值精度
BOSE_MAX_N = 500

# 模型 PMF 归一化所使用的词长支撑上界（离散分布归一化用）
PMF_SUPPORT_MAX = 200

# 词长过滤：剔除音节数<=0 或词频<=0 的记录
MIN_SYLLABLES = 1
MIN_FREQUENCY = 1


# ================================================================
# 5. 11 个模型的元信息（供 run_all 与汇总脚本使用）
# ================================================================
# model_id -> (中文名, 参数名列表)
MODEL_META = {
    "shifted_poisson": ("1-位移泊松", ["lambda"]),
    "gamma": ("Gamma", ["alpha", "beta"]),
    "poisson_mixture": ("混合泊松", ["pi", "lambda1", "lambda2"]),
    "zipf_alekseev": ("Zipf-Alekseev", ["a", "b"]),
    "negative_binomial": ("负二项", ["r", "p"]),
    "hyper_poisson": ("Hyper-Poisson", ["lambda", "b"]),
    "log_normal": ("对数正态", ["mu", "sigma"]),
    "bose": ("玻色子模型", ["d", "beta", "mu_tilde", "scale"]),
    "ehp": ("指数超泊松(EHP)", ["lambda", "b", "alpha"]),
    "generalized_gamma": ("广义伽马(GG)", ["a", "b", "p"]),
    "sichel": ("西切尔(Sichel)", ["theta", "sigma2", "gamma"]),
}

# 模型执行顺序
MODEL_ORDER = list(MODEL_META.keys())


if __name__ == "__main__":
    # 自检：打印关键路径
    print("BASE_DIR      =", BASE_DIR)
    print("PROJECT_ROOT  =", PROJECT_ROOT)
    print("CMUDICT_PATH  =", CMUDICT_PATH, "(exists:", os.path.exists(CMUDICT_PATH), ")")
    print("BNC_RAW_ROOT  =", BNC_RAW_ROOT, "(exists:", os.path.exists(BNC_RAW_ROOT), ")")
    print("模型数量      =", len(MODEL_META))
