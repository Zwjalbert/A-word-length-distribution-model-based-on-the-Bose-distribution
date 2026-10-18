# 03_拟合评价

模型拟合引擎、拟合优度指标与残差诊断。

## 模块清单

| 文件 | 功能 |
|------|------|
| `fitter.py` | curve_fit 封装：多初始值、边界、失败处理、置信区间 |
| `metrics.py` | R²/R²_adj、AIC/AICc/BIC、AIC权重、卡方、KS 检验 |
| `error_analysis.py` | 皮尔逊残差、异常点、卡方贡献分解 |

## 主要函数

### fitter.py
- `fit_model(model_id, ell, y_obs, total_count, dist_type, ...)` → 结构化拟合结果
- `DEFAULT_P0` / `DEFAULT_BOUNDS`：各模型默认初始值与边界

### metrics.py
- `compute_all_metrics(ell, y_obs, y_pred, n_params)` → 全部指标
- `aic_weights(aic_values)` → 模型相对权重
- `chi2_test(y_obs, y_pred, n_params)` → 卡方检验
- `ks_test(ell, y_obs, y_pred)` → KS 统计量与 p 值

### error_analysis.py
- `pearson_residuals(y_obs, y_pred)` → 标准化残差
- `identify_outliers(residuals, threshold=2.0)` → 异常 bin
- `chi2_contributions(y_obs, y_pred)` → 逐 bin 卡方贡献
- `residual_summary(ell, y_obs, y_pred)` → 诊断摘要

## 指标口径

- **对数似然**（泊松假设）：`ln L = Σ[y ln(ŷ) − ŷ − ln(y!)]`
- **AIC**：`2k − 2 ln L`
- **AICc**：`AIC + 2k(k+1)/(n−k−1)`
- **BIC**：`k ln n − 2 ln L`
- **AIC 权重**：`wᵢ = exp(−0.5·ΔAICᵢ) / Σ exp(−0.5·ΔAICⱼ)`

## 用法示例

```python
from fitter import fit_model
from metrics import compute_all_metrics

res = fit_model("gamma", ell, y_obs, total_count, dist_type="token")
m = compute_all_metrics(ell, y_obs, res["predicted"], res["n_params"])
print(m["R2_adj"], m["AIC"], m["KS_stat"])
```
