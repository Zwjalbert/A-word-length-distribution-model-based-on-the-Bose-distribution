# 词长分布量子模型实验代码

BNC Baby 四语料（学术 / 新闻 / 口语 / 小说）词长分布拟合实验的模块化代码。
对 **11 个候选模型**进行统一拟合、指标对比与可视化，核心为耦合语义场（玻色子）模型。

## 目录结构

```
实验代码/
├── 01_语料处理/          # 语料下载、解析、CMU 音节标注、分布计算
│   ├── bnc_baby_processor.py
│   ├── corpus_downloader.py
│   ├── corpus_processor.py
│   ├── config.py         # 全局配置（路径、绘图、拟合超参）
│   └── README.md
├── 02_模型定义/          # 11 个模型的 PMF
│   ├── models.py
│   └── README.md
├── 03_拟合评价/          # 拟合引擎与统计指标
│   ├── fitter.py
│   ├── metrics.py
│   ├── error_analysis.py
│   └── README.md
├── 04_图像绘制/          # 基础 / SCI英文 / SCI中文 三种绘图
│   ├── plotter.py
│   ├── plotter_sci.py
│   ├── plotter_sci_cn.py
│   └── README.md
├── 05_一键运行/          # 全流程编排
│   ├── run_all.py
│   └── README.md
├── data/
│   └── cmudict.dict     # CMU 发音词典
└── README.md             # 本文件
```

## 11 个模型

| # | 模型 | 参数数 |
|---|------|--------|
| 1 | 1-位移泊松 | 1 |
| 2 | Gamma | 2 |
| 3 | 混合泊松 | 3 |
| 4 | Zipf-Alekseev | 2 |
| 5 | 负二项 | 2 |
| 6 | Hyper-Poisson | 2 |
| 7 | 对数正态 | 2 |
| 8 | **玻色子模型**（d, β, μ̃, scale） | 4 |
| 9 | EHP 指数超泊松 | 3 |
| 10 | 广义伽马(Stacy) | 3 |
| 11 | Sichel 西切尔 | 3 |

### 玻色子模型
- **词例分布**（全数值积分）：`P(ℓ) ∝ ℓ^(d/2) · B(d/2, β(ℓ-1)+μ̃)`
- **词型分布**（低温近似）：`P(ℓ) ∝ ℓ^(d/2) · exp(-βℓ+μ̃)`

## 环境依赖

```
numpy, scipy, pandas, matplotlib, openpyxl
```

## 快速开始

```bash
# 1. 一键运行全部实验（自动处理语料、拟合、出图）
cd 05_一键运行
python run_all.py

# 2. 或单步调试
python run_all.py --corpus academic --dist token
```

## 数据路径

- **只读输入**（不随仓库分发，在 `config.BNC_RAW_ROOT` 配置）：
  `...\实验0914\英文原始语料\download\Texts\{aca,news,dem,fic}\*.xml`
- **CMU 词典**：`data/cmudict.dict`（已随仓库分发）
- **输出**：`<项目根>/results/<语料>/`

## 拟合与指标口径

- 拟合：`scipy.optimize.curve_fit`，多初始值 + 参数边界 + 失败重试
- 似然：泊松对数似然；AIC/AICc/BIC/AIC 权重
- 检验：卡方（合并小期望 bin）、Kolmogorov-Smirnov
- 评价：R²、R²_adj、RMSE、MAE、MAPE、标准化残差

## 代码来源

由单体脚本模块化重构而来：
- 玻色子模型 / 玻色积分：`词长分布1.2.py`
- 拟合指标 / 多初始值：`V2.0.py`
- 模型参数结构：`BNC实验结果/拟合结果/*/*_results.json`
