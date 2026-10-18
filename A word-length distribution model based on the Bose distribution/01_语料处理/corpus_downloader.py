# -*- coding: utf-8 -*-
"""
================================================================
corpus_downloader.py —— 通用语料下载与解压工具
================================================================

功能概述：
    提供从网络下载 BNC Baby（或其他公开 XML 语料）压缩包、
    解压、校验完整性的通用工具函数。

典型场景：
    首次运行实验时，若本地 BNC Baby 语料不存在，可调用
    :func:`download_bnc_baby` 自动下载并解压到指定目录。

说明：
    BNC Baby 的官方分发地址为 Oxford Text Archive /
    NLTK 数据源。本模块仅提供下载骨架，URL 与校验码
    可按需更新；已手动下载好的语料可直接跳过下载步骤。
"""

import os
import shutil
import hashlib
import zipfile
import tarfile
import urllib.request
from typing import Optional

# 复用全局配置
try:
    from config import BNC_RAW_ROOT, ensure_dir
except ImportError:  # 允许以脚本方式直接运行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import BNC_RAW_ROOT, ensure_dir


# BNC Baby 常见下载地址（官方镜像，按需替换为可用 URL）
BNC_BABY_URLS = [
    "https://nltk.org/nltk_data/packages/corpora/bnc.zip",
    "https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554",
]


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """计算文件的 SHA-256 摘要，用于下载完整性校验。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, save_path: str, expected_sha256: Optional[str] = None) -> str:
    """
    从指定 URL 下载文件并保存到本地。

    参数：
        url:               下载链接
        save_path:         本地保存路径（含文件名）
        expected_sha256:   可选，预期 SHA-256 校验和；提供则下载后校验

    返回：
        保存后的本地绝对路径

    异常：
        下载失败或校验不通过时抛出 RuntimeError
    """
    ensure_dir(os.path.dirname(save_path))
    print(f"[下载] {url}\n       -> {save_path}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            print(f"\r       进度: {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=_progress)
        print()  # 换行
    except Exception as e:  # noqa: BLE001 - 下载失败需包装后抛出
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(f"下载失败: {e}") from e

    if expected_sha256 is not None:
        actual = _sha256_of_file(save_path)
        if actual.lower() != expected_sha256.lower():
            os.remove(save_path)
            raise RuntimeError(
                f"校验失败: 期望 {expected_sha256}, 实际 {actual}"
            )
        print("       校验通过 ✓")
    return save_path


def extract_archive(archive_path: str, dest_dir: str) -> str:
    """
    解压 zip / tar.gz 压缩包到目标目录。

    参数：
        archive_path: 压缩包路径
        dest_dir:     解压目标目录

    返回：
        解压后的根目录路径
    """
    ensure_dir(dest_dir)
    print(f"[解压] {archive_path} -> {dest_dir}")
    if archive_path.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.lower().endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path}")
    return dest_dir


def corpus_exists(root: Optional[str] = None) -> bool:
    """
    检查 BNC Baby 语料是否已在本地就绪。

    判定标准：root/Texts/ 目录下存在 aca、news、dem、fic 四个子目录。
    """
    root = root or BNC_RAW_ROOT
    texts = os.path.join(root, "Texts")
    if not os.path.isdir(texts):
        return False
    return all(os.path.isdir(os.path.join(texts, d))
               for d in ("aca", "news", "dem", "fic"))


def download_bnc_baby(dest_root: Optional[str] = None) -> str:
    """
    一站式下载并解压 BNC Baby 语料。

    参数：
        dest_root: 语料根目录，默认使用 config.BNC_RAW_ROOT

    返回：
        语料根目录路径
    """
    dest_root = dest_root or BNC_RAW_ROOT
    if corpus_exists(dest_root):
        print(f"[跳过] BNC Baby 语料已存在于 {dest_root}")
        return dest_root

    ensure_dir(dest_root)
    archive_path = os.path.join(dest_root, "bnc_baby.zip")

    last_err = None
    for url in BNC_BABY_URLS:
        try:
            download_file(url, archive_path)
            extract_archive(archive_path, dest_root)
            os.remove(archive_path)
            if corpus_exists(dest_root):
                print(f"[完成] BNC Baby 语料就绪: {dest_root}")
                return dest_root
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[警告] 镜像 {url} 不可用: {e}")

    raise RuntimeError(f"所有下载镜像均失败。最后错误: {last_err}\n"
                       f"请手动下载 BNC Baby 并解压到: {dest_root}/Texts/")


if __name__ == "__main__":
    # 命令行自检：检查语料是否就绪
    if corpus_exists():
        print("BNC Baby 语料已就绪:", BNC_RAW_ROOT)
    else:
        print("BNC Baby 语料未就绪，可调用 download_bnc_baby() 下载。")
