# 05_一键运行

编排「语料 → 分布类型 → 模型」的完整实验流程。

## 文件

- `run_all.py`：主入口，一键执行全部实验。

## 用法

```bash
# 全量运行（四语料 × token/type × 11 模型）
cd 05_一键运行
python run_all.py

# 单步调试
python run_all.py --corpus academic --dist token
```

## 流程

1. **语料处理**：若 `data/processed/` 下词表缺失，自动解析 BNC XML 并 CMU 标注。
2. **分布计算**：对每个语料计算词例/词型词长分布。
3. **模型拟合**：对每个分布依次拟合 11 个模型。
4. **指标计算**：R²_adj、AIC、AIC 权重、KS 检验等。
5. **结果落盘**：
   - `<corpus>/bnc_<corpus>_<dist>_comparison.csv`（模型对比表）
   - `<corpus>/bnc_<corpus>_<dist>_results.json`（完整结果含观测/预测/参数）
   - `<corpus>/bnc_<corpus>_<dist>_bose_fit.png` / `_bose_res.png`
   - `<corpus>/bnc_<corpus>_<dist>_top5.png` / `_r2bar.png`

## 主要函数

- `main(corpus_filter, dist_filter)`：主入口
- `run_single_distribution(corpus_key, dist_type, csv, out_dir)`：单分布全模型拟合
- `save_distribution_outputs(...)`：保存 CSV/JSON/图表
