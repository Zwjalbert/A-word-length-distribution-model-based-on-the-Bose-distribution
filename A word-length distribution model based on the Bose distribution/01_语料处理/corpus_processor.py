# -*- coding: utf-8 -*-
"""
================================================================
corpus_processor.py —— 语料清洗与词长分布计算
================================================================

功能概述：
    读取「词目 | 词频 | 音节数」词表 CSV，计算两种核心分布：
        1. 词例分布 (token distribution)：各词长的总出现频次
           N_token(ℓ) = Σ_{词目音节数=ℓ} frequency
        2. 词型分布 (type distribution)：各词长的不同词目数
           N_type(ℓ)  = #{词目: 音节数=ℓ}

同时计算泊松误差 √N（计数数据的标准误）与总计数。

输入 CSV 列：word, frequency, syllables
输出：ell（词长数组）、N_token、N_type 及对应误差。
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from config import MIN_SYLLABLES, MIN_FREQUENCY
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import MIN_SYLLABLES, MIN_FREQUENCY


def load_wordlist(csv_path: str) -> pd.DataFrame:
    """
    加载词表 CSV，并做基本清洗。

    清洗规则：
        - 剔除 syllables <= 0 的记录
        - 剔除 frequency <= 0 的记录
        - 词目转小写

    参数：
        csv_path: bnc_<corpus>_wordlist.csv 路径

    返回：
        清洗后的 DataFrame，含列 word/frequency/syllables
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[(df["syllables"] >= MIN_SYLLABLES) &
            (df["frequency"] >= MIN_FREQUENCY)]
    df["word"] = df["word"].astype(str).str.lower()
    return df.reset_index(drop=True)


def compute_length_distributions(df: pd.DataFrame) -> Dict:
    """
    由词表 DataFrame 计算词长分布。

    参数：
        df: load_wordlist 返回的 DataFrame

    返回：
        dict 含以下键：
            ell          (np.ndarray): 词长（音节数）升序数组
            N_token_obs  (np.ndarray): 各词长总词例数
            N_token_err  (np.ndarray): 词例数标准误 √N
            N_type_obs   (np.ndarray): 各词型数
            N_type_err   (np.ndarray): 词型数标准误 √N
            N_total_tokens (int): 总词例数
            N_total_types  (int): 总词型数
    """
    N_total_tokens = int(df["frequency"].sum())
    N_total_types = int(len(df))

    # 词例分布：按音节数对词频求和
    token_dist = (df.groupby("syllables")["frequency"]
                  .sum().reset_index())
    # 词型分布：按音节数计数
    type_dist = (df.groupby("syllables").size()
                 .reset_index(name="count"))

    ell = np.array(token_dist["syllables"], dtype=float)
    N_token_obs = np.array(token_dist["frequency"], dtype=float)
    N_type_obs = np.array(type_dist["count"], dtype=float)

    # 泊松计数误差 σ = √N
    N_token_err = np.sqrt(N_token_obs)
    N_type_err = np.sqrt(N_type_obs)

    return {
        "ell": ell,
        "N_token_obs": N_token_obs,
        "N_token_err": N_token_err,
        "N_type_obs": N_type_obs,
        "N_type_err": N_type_err,
        "N_total_tokens": N_total_tokens,
        "N_total_types": N_total_types,
    }


def load_distribution_from_csv(csv_path: str) -> Dict:
    """
    一站式：读词表 CSV -> 清洗 -> 计算分布。

    参数：
        csv_path: 词表 CSV 路径

    返回：
        compute_length_distributions 的结果字典
    """
    df = load_wordlist(csv_path)
    result = compute_length_distributions(df)
    print(f"[分布] {os.path.basename(csv_path)}: "
          f"词长 {int(result['ell'].min())}~{int(result['ell'].max())}, "
          f"词例 {result['N_total_tokens']:,}, "
          f"词型 {result['N_total_types']:,}")
    return result


def split_by_distribution(result: Dict, dist_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    从分布结果中取出指定分布类型的拟合用数据。

    参数：
        result:     compute_length_distributions 返回的字典
        dist_type:  "token" 或 "type"

    返回：
        (ell, y_obs, y_err, N_total)
    """
    if dist_type == "token":
        return (result["ell"], result["N_token_obs"],
                result["N_token_err"], result["N_total_tokens"])
    elif dist_type == "type":
        return (result["ell"], result["N_type_obs"],
                result["N_type_err"], result["N_total_types"])
    raise ValueError(f"未知分布类型: {dist_type}（应为 token 或 type）")


if __name__ == "__main__":
    # 自检：处理学术语料并打印分布
    from config import CORPUS_OUT_DIR
    sample = os.path.join(CORPUS_OUT_DIR, "bnc_academic_wordlist.csv")
    if os.path.exists(sample):
        r = load_distribution_from_csv(sample)
        for e, t in zip(r["ell"], r["N_token_obs"]):
            print(f"  ℓ={int(e):2d}: token={int(t):8d}")
    else:
        print(f"未找到 {sample}，请先运行 bnc_baby_processor.py")
