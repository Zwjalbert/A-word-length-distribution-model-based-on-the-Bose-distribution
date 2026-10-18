# -*- coding: utf-8 -*-
"""
================================================================
run_all.py —— 一键执行全流程
================================================================

按「语料 → 分布类型 → 模型」的顺序执行完整实验：

    1. 处理 BNC Baby 四语料（academic/news/spoken/fiction）
       生成「词目|词频|音节数」词表；
    2. 计算词例/词型两种词长分布；
    3. 对每个分布拟合全部 11 个候选模型；
    4. 计算全套拟合指标（R²_adj / AIC / AIC权重 / KS 等）；
    5. 输出对比 CSV、结果 JSON 与图表到 results/ 目录。

用法：
    cd 05_一键运行
    python run_all.py                 # 全量运行
    python run_all.py --corpus academic --dist token   # 单步调试
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ---- 把各模块目录加入 sys.path ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _sub in ["01_语料处理", "02_模型定义", "03_拟合评价", "04_图像绘制"]:
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (CORPUS_MAP, CORPUS_CN, DIST_TYPES, MODEL_META,
                    MODEL_ORDER, CORPUS_OUT_DIR, RESULT_ROOT, ensure_dir)
from bnc_baby_processor import process_all_corpora
from corpus_processor import (load_distribution_from_csv,
                              split_by_distribution)
from fitter import fit_model
from metrics import (compute_all_metrics, aic_weights, bic_weights)
from error_analysis import residual_summary
from plotter import (plot_fit_comparison, plot_residuals,
                     plot_multi_model_fit, plot_metrics_bar)


# ================================================================
# 1. 单个 (语料, 分布类型) 的完整拟合流程
# ================================================================
def run_single_distribution(corpus_key, dist_type, wordlist_csv, out_dir):
    """
    对单个语料的单个分布类型，拟合全部 11 个模型。

    参数：
        corpus_key:   academic/news/spoken/fiction
        dist_type:    token/type
        wordlist_csv: 该语料词表 CSV 路径
        out_dir:      该 (语料) 结果输出目录

    返回：
        (comparison_df, results_dict)
    """
    dist = load_distribution_from_csv(wordlist_csv)
    ell, y_obs, y_err, total_count = split_by_distribution(dist, dist_type)

    print(f"\n{'='*60}")
    print(f"语料: {CORPUS_CN[corpus_key]}  |  分布: {dist_type}  |  "
          f"总计数: {total_count:,}")
    print(f"{'='*60}")

    results = {}
    rows = []
    for model_id in MODEL_ORDER:
        name_cn = MODEL_META[model_id][0]
        print(f" 拟合 {model_id:20s} ...", end=" ")
        fit = fit_model(model_id, ell, y_obs, total_count, dist_type=dist_type)
        if not fit["fit_success"]:
            print(f"失败: {fit.get('error','')}")
            continue
        y_pred = np.array(fit["predicted"])
        metrics = compute_all_metrics(ell, y_obs, y_pred, fit["n_params"])
        res = residual_summary(ell, y_obs, y_pred)

        fit["model_name"] = name_cn
        fit["distribution_type"] = dist_type
        fit["metrics"] = metrics
        fit["residual_summary"] = res
        results[model_id] = fit

        # 对比表行
        row = {
            "model_id": model_id,
            "model_name": name_cn,
            "n_params": fit["n_params"],
            "fit_success": True,
            "R2": metrics["R2"],
            "R2_adj": metrics["R2_adj"],
            "chi2": metrics["chi2"],
            "chi2_N": metrics["chi2_N"],
            "chi2_df": metrics["chi2_df"],
            "chi2_pvalue": metrics["chi2_pvalue"],
            "log_likelihood": metrics["log_likelihood"],
            "AIC": metrics["AIC"],
            "AICc": metrics["AICc"],
            "BIC": metrics["BIC"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "KS_stat": metrics["KS_stat"],
            "KS_pvalue": metrics["KS_pvalue"],
        }
        # 展开参数列为 param_<name>
        for pname, pval in fit["param_dict"].items():
            row[f"param_{pname}"] = pval
        rows.append(row)
        print(f"R2_adj={metrics['R2_adj']:.4f}  AIC={metrics['AIC']:.1f}")

    comparison = pd.DataFrame(rows)
    if len(comparison) > 0:
        comparison["AIC_weight"] = aic_weights(comparison["AIC"].values)
        comparison["BIC_weight"] = bic_weights(comparison["BIC"].values)

    return comparison, results, dist, (ell, y_obs, y_err, total_count)


# ================================================================
# 2. 结果落盘
# ================================================================
def save_distribution_outputs(corpus_key, dist_type, comparison, results,
                              dist, arrays, out_dir):
    """保存对比 CSV、完整 JSON 与图表。"""
    prefix = f"bnc_{corpus_key}_{dist_type}"
    sub_dir = ensure_dir(os.path.join(out_dir, corpus_key))

    # 2.1 对比 CSV
    csv_path = os.path.join(sub_dir, f"{prefix}_comparison.csv")
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 2.2 完整结果 JSON（含观测/预测/参数）
    json_path = os.path.join(sub_dir, f"{prefix}_results.json")
    serializable = {
        "distribution_type": dist_type,
        "n_data_points": int(len(arrays[0])),
        "total_count": int(arrays[3]),
        "results": results,
        "aic_values": comparison.set_index("model_id")["AIC"].to_dict(),
        "aic_weights": comparison.set_index("model_id")["AIC_weight"].to_dict(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2,
                  default=_json_default)

    # 2.3 图表：玻色子模型拟合对比 + 残差
    if "bose" in results:
        bose = results["bose"]
        ell = np.array(bose["lengths"])
        plot_fit_comparison(
            ell, np.array(bose["observed"]), np.array(bose["predicted"]),
            y_err=np.sqrt(np.array(bose["observed"])),
            title=f"{CORPUS_CN[corpus_key]} {dist_type} 玻色子模型拟合",
            save_path=os.path.join(sub_dir, f"{prefix}_bose_fit.png"))
        resd = np.array(bose["metrics"]["std_residuals"])
        plot_residuals(ell, resd,
                       title=f"{CORPUS_CN[corpus_key]} {dist_type} 玻色子残差",
                       save_path=os.path.join(sub_dir, f"{prefix}_bose_res.png"))

    # 2.4 多模型叠加图（按 AIC 取前 4 + 强制包含玻色子模型）
    if len(comparison) >= 2:
        top = comparison.sort_values("AIC").head(4)
        # 强制加入玻色子模型（无论排名）
        bose_row = comparison[comparison["model_id"] == "bose"]
        if len(bose_row) > 0 and "bose" not in top["model_id"].values:
            top = pd.concat([top, bose_row], ignore_index=True)
        curves = [(r["model_name"],
                   np.array(results[r["model_id"]]["predicted"]))
                  for _, r in top.iterrows()]
        ell = np.array(next(iter(results.values()))["lengths"])
        yobs = np.array(next(iter(results.values()))["observed"])
        plot_multi_model_fit(
            ell, yobs, curves,
            title=f"{CORPUS_CN[corpus_key]} {dist_type}  多模型对比（含玻色子）",
            save_path=os.path.join(sub_dir, f"{prefix}_top5.png"))
        plot_metrics_bar(
            comparison["model_name"].tolist(),
            comparison["R2_adj"].tolist(),
            metric_label="R²_adj",
            title=f"{CORPUS_CN[corpus_key]} {dist_type}  R²_adj",
            save_path=os.path.join(sub_dir, f"{prefix}_r2bar.png"))

    return csv_path, json_path


def _json_default(o):
    """JSON 序列化兜底：转 numpy 类型。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ================================================================
# 3. 主入口
# ================================================================
def main(corpus_filter=None, dist_filter=None):
    """
    全流程主入口。

    参数：
        corpus_filter: 仅处理指定语料（None=全部四个）
        dist_filter:   仅处理指定分布（None=token+type）
    """
    print("=" * 70)
    print("词长分布量子模型实验 —— 一键运行")
    print("=" * 70)

    # 步骤 1：处理语料（若词表已存在则直接复用）
    ensure_dir(CORPUS_OUT_DIR)
    ensure_dir(RESULT_ROOT)
    wordlists = {}
    for key in CORPUS_MAP:
        csv_path = os.path.join(CORPUS_OUT_DIR, f"bnc_{key}_wordlist.csv")
        if not os.path.exists(csv_path):
            print(f"\n[语料] {key} 词表缺失，开始解析 ...")
            from bnc_baby_processor import process_corpus, load_cmudict
            smap = load_cmudict()
            process_corpus(key, smap)
        wordlists[key] = csv_path

    # 步骤 2：逐语料、逐分布、逐模型拟合
    corpora = [corpus_filter] if corpus_filter else list(CORPUS_MAP)
    dists = [dist_filter] if dist_filter else DIST_TYPES

    for corpus_key in corpora:
        for dist_type in dists:
            comparison, results, dist, arrays = run_single_distribution(
                corpus_key, dist_type, wordlists[corpus_key], RESULT_ROOT)
            if len(comparison) == 0:
                print(f"[警告] {corpus_key}/{dist_type} 无成功拟合")
                continue
            paths = save_distribution_outputs(
                corpus_key, dist_type, comparison, results,
                dist, arrays, RESULT_ROOT)
            print(f"  已保存: {paths[0]}")
            print(f"          {paths[1]}")

    print("\n" + "=" * 70)
    print("拟合阶段完成，开始绘制论文插图 ...")
    print("=" * 70)

    # 步骤3：绘制论文SCI风格插图（5张）
    # 注：import * 不允许出现在函数内部（SyntaxError），改为导入模块；
    # 模块 plot_paper_figures 在模块级执行绘图，导入即触发图1–图5 绘制。
    try:
        import plot_paper_figures as _ppf  # noqa
        print("\n论文插图绘制完成！")
    except Exception as e:
        print(f"[警告] 论文插图绘制失败: {e}")
        print("可手动运行: python 04_图像绘制/plot_paper_figures.py")

    print("\n" + "=" * 70)
    print(f"全部完成！结果目录: {RESULT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="词长分布模型拟合一键运行")
    parser.add_argument("--corpus", default=None,
                        help="指定语料 academic/news/spoken/fiction")
    parser.add_argument("--dist", default=None,
                        help="指定分布类型 token/type")
    args = parser.parse_args()
    main(corpus_filter=args.corpus, dist_filter=args.dist)
