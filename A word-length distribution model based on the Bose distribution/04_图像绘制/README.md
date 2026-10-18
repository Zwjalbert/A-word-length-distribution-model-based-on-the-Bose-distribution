# 04_图像绘制

实验结果可视化，提供三种版式以适应不同场景。

## 模块清单

| 文件 | 用途 | 语言 |
|------|------|------|
| `plotter.py` | 基础绘图：拟合对比、残差、多模型叠加、指标柱状图 | 中文 |
| `plotter_sci.py` | SCI 期刊英文版式（Arial，紧凑，单栏） | 英文 |
| `plotter_sci_cn.py` | SCI 风格中文版式（黑体，紧凑，会议海报） | 中文 |

## 主要函数

### plotter.py
- `plot_fit_comparison(ell, y_obs, y_pred, y_err, save_path)`
- `plot_residuals(ell, residuals, save_path)`
- `plot_multi_model_fit(ell, y_obs, model_curves, save_path)`
- `plot_metrics_bar(model_names, metric_values, save_path)`

### plotter_sci.py
- `apply_sci_style()`
- `plot_fit_sci(...)` / `plot_multi_fit_sci(...)` / `plot_aic_rank_sci(...)`

### plotter_sci_cn.py
- `apply_sci_cn_style()`
- `plot_fit_sci_cn(...)` / `plot_two_panel_cn(...)` / `plot_aic_rank_cn(...)`

## 用法示例

```python
from plotter_sci import plot_fit_sci
plot_fit_sci(ell, y_obs, y_pred,
             save_path="results/academic/token_fit.png")
```

## 输出规范

- 所有图 300 dpi，PNG 格式
- `bbox_inches="tight"` 自动裁边
- 保存前自动 `ensure_dir` 创建目录
