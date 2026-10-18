# 01_语料处理

BNC Baby 语料的下载、解析、CMU 音节标注与词长分布计算。

## 模块清单

| 文件 | 功能 |
|------|------|
| `config.py` | 全局配置：路径、绘图 rcParams、拟合超参数、模型元信息 |
| `corpus_downloader.py` | 通用语料下载/解压/校验工具 |
| `bnc_baby_processor.py` | BNC Baby XML 解析 + CMU 音节标注，输出词表 CSV |
| `corpus_processor.py` | 词表清洗、词例/词型词长分布计算 |

## 主要函数

### config.py
- `apply_plot_style()`：应用科研绘图风格
- `ensure_dir(path)`：递归创建目录
- 常量：`BNC_RAW_ROOT`、`CORPUS_MAP`、`MODEL_META`、`FIT_MAXFEV` 等

### bnc_baby_processor.py
- `load_cmudict(dict_path)` → `{词目: 音节数}`
- `parse_bnc_xml(xml_path)` → 词目列表
- `process_corpus(corpus_key, syllable_map)` → 词表 CSV
- `process_all_corpora()` → 一次处理四语料

### corpus_processor.py
- `load_wordlist(csv_path)` → 清洗后 DataFrame
- `compute_length_distributions(df)` → `ell / N_token / N_type` 及误差
- `load_distribution_from_csv(csv_path)` → 一站式
- `split_by_distribution(result, "token"|"type")` → 拟合用数组

## 用法示例

```python
# 1) 处理全部语料（首次）
from bnc_baby_processor import process_all_corpora
process_all_corpora()

# 2) 计算某语料词长分布
from corpus_processor import load_distribution_from_csv, split_by_distribution
dist = load_distribution_from_csv("data/processed/bnc_academic_wordlist.csv")
ell, y_obs, y_err, N = split_by_distribution(dist, "token")
```

## 目录映射

BNC Baby 原始子目录与逻辑语料名的对应（见 `config.CORPUS_MAP`）：

| 逻辑名 | XML 子目录 | 含义 |
|--------|-----------|------|
| academic | aca | 学术书面语 |
| news | news | 新闻书面语 |
| spoken | dem | 口语对话 |
| fiction | fic | 小说文学 |
