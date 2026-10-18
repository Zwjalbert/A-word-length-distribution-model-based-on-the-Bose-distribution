# -*- coding: utf-8 -*-
"""
================================================================
bnc_baby_processor.py —— BNC Baby 语料解析与 CMU 音节标注
================================================================

功能概述：
    1. 读取 BNC Baby 原始 XML 文件（<w hw="词目">词形</w>）
    2. 加载 CMU 发音词典，为每个词目计算音节数
    3. 输出「词目 | 词频 | 音节数」的词表 CSV，供后续词长分布统计

BNC Baby 目录结构：
    <BNC_RAW_ROOT>/Texts/{aca,news,dem,fic}/*.xml
    其中：
        aca  -> academic 学术书面语
        news -> news     新闻书面语
        dem  -> spoken   口语对话
        fic  -> fiction  小说文学

音节计数规则：
    CMU 词典中每个元音音素携带重音标记（0/1/2），
    音节数 = 以数字结尾的音素个数。
    词典未收录的词目按字母长度的启发式规则估算，
    或标记为 0 并在下游过滤。
"""

import os
import re
import csv
from collections import Counter
from typing import Dict, List, Tuple

try:
    from config import (
        BNC_RAW_ROOT, CORPUS_MAP, CMUDICT_PATH,
        CORPUS_OUT_DIR, ensure_dir, MIN_FREQUENCY,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        BNC_RAW_ROOT, CORPUS_MAP, CMUDICT_PATH,
        CORPUS_OUT_DIR, ensure_dir, MIN_FREQUENCY,
    )


# ================================================================
# 1. CMU 发音词典加载
# ================================================================
def load_cmudict(dict_path: str = None) -> Dict[str, int]:
    """
    加载 CMU 发音词典，构建 {小写词目: 音节数} 映射。

    音节数 = 音素中以数字（重音标记 0/1/2）结尾的音素个数。
    多发音词目（如 "read(1)"）取首次出现的发音。

    参数：
        dict_path: cmudict.dict 路径，默认使用 config.CMUDICT_PATH

    返回：
        dict: {word_lower: syllable_count}
    """
    dict_path = dict_path or CMUDICT_PATH
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"CMU 词典未找到: {dict_path}")

    syllable_map: Dict[str, int] = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            # 忽略多发音变体标记 read(1)，只保留首个发音
            if "(" in word:
                word = re.sub(r"\(\d+\)", "", word)
            if word in syllable_map:
                continue
            phonemes = parts[1:]
            # 音节数 = 携带重音数字的音素个数
            n_syl = sum(1 for ph in phonemes if ph[-1].isdigit())
            syllable_map[word] = n_syl if n_syl > 0 else 1
    return syllable_map


def estimate_syllables_fallback(word: str) -> int:
    """
    词典未收录词目的音节数启发式估算（基于元音簇计数）。

    规则：按连续元音簇（aeiouy）计数，末尾静默 e 不计。
    """
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    # 末尾 "e" 单独成簇且前面有其他元音时视为静默 e
    if word.endswith("e") and n > 1 and groups[-1] == "e":
        n -= 1
    return max(1, n)


# ================================================================
# 2. BNC XML 解析
# ================================================================
# 匹配 <w ... hw="lemma" ...>词形</w>
_W_RE = re.compile(
    r'<w\b[^>]*\bhw="([^"]*)"[^>]*>(.*?)</w>',
    re.IGNORECASE | re.DOTALL,
)
# 提取标签内纯文本（去除嵌套标签）
_INNER_TAG_RE = re.compile(r"<[^>]+>")


def parse_bnc_xml(xml_path: str) -> List[str]:
    """
    解析单个 BNC XML 文件，提取词目（lemma）列表。

    仅保留 pos 不为标点/UNC 的词元，使用 hw 属性作为词目。

    参数：
        xml_path: XML 文件路径

    返回：
        list[str]: 词目列表（已小写化、清洗）
    """
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lemmas: List[str] = []
    for m in _W_RE.finditer(content):
        hw = m.group(1).strip().lower()
        # 跳过纯标点、占位符
        if not hw or not re.search(r"[a-z]", hw):
            continue
        lemmas.append(hw)
    return lemmas


def process_corpus(corpus_key: str,
                   syllable_map: Dict[str, int],
                   raw_root: str = None,
                   out_dir: str = None) -> str:
    """
    处理单个语料子集（academic/news/spoken/fiction），
    统计词频并标注音节数，输出词表 CSV。

    参数：
        corpus_key: 逻辑语料名（academic/news/spoken/fiction）
        syllable_map: CMU 词典音节映射
        raw_root:   BNC 语料根目录
        out_dir:    词表 CSV 输出目录

    返回：
        输出 CSV 的路径
    """
    raw_root = raw_root or BNC_RAW_ROOT
    out_dir = out_dir or CORPUS_OUT_DIR
    subdir = CORPUS_MAP[corpus_key]
    texts_dir = os.path.join(raw_root, "Texts", subdir)
    if not os.path.isdir(texts_dir):
        raise FileNotFoundError(f"语料目录不存在: {texts_dir}")

    # 1) 遍历该子集所有 XML，统计词目频次
    freq_counter: Counter = Counter()
    xml_files = [f for f in os.listdir(texts_dir) if f.lower().endswith(".xml")]
    print(f"[{corpus_key}] 解析 {len(xml_files)} 个 XML 文件 ...")
    for fname in xml_files:
        try:
            lemmas = parse_bnc_xml(os.path.join(texts_dir, fname))
            freq_counter.update(lemmas)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] {fname} 解析失败: {e}")

    # 2) 标注音节数，写出 CSV
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"bnc_{corpus_key}_wordlist.csv")
    n_unknown = 0
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency", "syllables"])
        for word, freq in sorted(freq_counter.items(), key=lambda x: -x[1]):
            if freq < MIN_FREQUENCY:
                continue
            syl = syllable_map.get(word)
            if syl is None:
                syl = estimate_syllables_fallback(word)
                n_unknown += 1
            if syl <= 0:
                continue
            writer.writerow([word, freq, syl])

    n_types = sum(1 for _ in open(out_path, encoding="utf-8-sig")) - 1
    print(f"[{corpus_key}] 完成: {n_types} 词型, "
          f"词典未收录(启发式估算) {n_unknown} 词, -> {out_path}")
    return out_path


def process_all_corpora(raw_root: str = None, out_dir: str = None) -> Dict[str, str]:
    """
    一次性处理全部四个 BNC Baby 语料子集。

    返回：
        dict: {corpus_key: csv_path}
    """
    print("=" * 60)
    print("加载 CMU 发音词典 ...")
    smap = load_cmudict()
    print(f"词典词条数: {len(smap)}")
    print("=" * 60)

    results = {}
    for key in CORPUS_MAP:
        results[key] = process_corpus(key, smap, raw_root, out_dir)
    return results


if __name__ == "__main__":
    paths = process_all_corpora()
    print("\n全部语料处理完成:")
    for k, v in paths.items():
        print(f"  {k:10s} -> {v}")
