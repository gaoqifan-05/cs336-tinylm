"""
数据加载 —— WikiText-2 预训练数据管道

流程:
    WikiText-2 原始文本
        → BPE 分词器 (data/tokenizer.json)
        → token id 长序列
        → 按 max_seq_len 切成等长块
        → PyTorch DataLoader

第一次运行会下载 WikiText-2 (~5MB) 并 tokenize（可能要几分钟），
结果缓存到 data/ 目录，之后直接加载缓存，秒级完成。
"""

import os
import sys
from typing import List

# 确保能 import src.tokenizer（无论从哪里运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import load_tokenizer


# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_CACHE = os.path.join(DATA_DIR, "train_ids.pt")
VAL_CACHE = os.path.join(DATA_DIR, "val_ids.pt")


class TokenDataset(Dataset):
    """把 token 序列切成 (输入, 目标) 对：输入[0:T] 预测 目标[0:T]（右移一位）"""

    def __init__(self, ids: List[int], seq_len: int):
        self.seq_len = seq_len
        # 切成 seq_len+1 的块（多出的 1 个 token 用来做 targets 的移位）
        n_blocks = len(ids) // (seq_len + 1)
        self.ids = torch.tensor(ids[: n_blocks * (seq_len + 1)], dtype=torch.long)
        self.ids = self.ids.view(-1, seq_len + 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        block = self.ids[idx]
        x = block[:-1]          # 输入：前 seq_len 个 token
        y = block[1:]           # 目标：后 seq_len 个 token（右移一位）
        return x, y


def _download_wikitext2() -> dict:
    """从 hf-mirror 镜像下载 WikiText-2 原始文本，返回 {'train': [...], 'validation': [...]}"""
    import pandas as pd
    import requests

    base = "https://hf-mirror.com/datasets/wikitext/resolve/main/wikitext-2-raw-v1"
    os.makedirs(DATA_DIR, exist_ok=True)

    result = {}
    for split, fname in [("train", "train-00000-of-00001.parquet"),
                         ("validation", "validation-00000-of-00001.parquet"),
                         ("test", "test-00000-of-00001.parquet")]:
        # 兼容两种命名：手动的 train.parquet 或原始 train-00000-of-00001.parquet
        local = os.path.join(DATA_DIR, f"{split}.parquet")
        raw_local = os.path.join(DATA_DIR, fname)
        if os.path.exists(raw_local) and not os.path.exists(local):
            os.rename(raw_local, local)
            print(f"[data] 重命名 {fname} → {split}.parquet")

        if not os.path.exists(local):
            url = f"{base}/{fname}"
            print(f"[data] 下载 {fname} ...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
            print(f"[data] 已保存 {local} ({len(r.content)/1e6:.1f} MB)")
        else:
            print(f"[data] 使用本地缓存 {local}")

        df = pd.read_parquet(local)
        # WikiText-2 parquet 的列是 "text"，过滤空行
        lines = [t for t in df["text"].tolist() if t.strip()]
        result[split] = lines
        print(f"[data] {split} 行数: {len(lines)}")

    return result


def _load_wikitext103(subset_ratio: float = 1.0, subset_mode: str = "first", subset_index: int = 0, seed: int = 42) -> dict:
    """从本地读取 WikiText-103 分片 parquet，返回 {'train': [...], 'validation': [...], 'test': [...]}

    subset_ratio: 取多少比例的训练数据（1.0=全量）
    subset_mode:  子集采样方式
        - "first"  取前 subset_ratio 比例（默认，向后兼容）
        - "random" 随机均匀采样 subset_ratio 比例（固定 seed 可复现）
        - "block"  取第 subset_index 个等宽分块（如 subset_ratio=0.1, index=1 → 取 10%~20% 这段）
    subset_index: block 模式下取第几块（从 0 开始）
    seed:         random 模式的随机种子
    """
    import pandas as pd
    import random

    os.makedirs(DATA_DIR, exist_ok=True)
    result = {}

    # 训练集：两个分片
    train_files = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.startswith("wt103_train") and f.endswith(".parquet")
    ])
    if not train_files:
        raise FileNotFoundError("未找到 WikiText-103 训练分片，请先下载 wt103_train-*.parquet 到 data/")

    train_lines = []
    for f in train_files:
        df = pd.read_parquet(f)
        train_lines.extend([t for t in df["text"].tolist() if t.strip()])
    # 取子集（按指定方式）
    if subset_ratio < 1.0:
        n_total = len(train_lines)
        n = int(n_total * subset_ratio)
        if subset_mode == "random":
            rng = random.Random(seed)
            train_lines = rng.sample(train_lines, n)
            print(f"[data] 使用训练子集: 随机 {subset_ratio*100:.0f}% = {n} 行 (seed={seed})")
        elif subset_mode == "block":
            start = int(n_total * subset_ratio * subset_index)
            end = int(n_total * subset_ratio * (subset_index + 1))
            train_lines = train_lines[start:end]
            pct_lo = subset_ratio * subset_index * 100
            pct_hi = subset_ratio * (subset_index + 1) * 100
            print(f"[data] 使用训练子集: 第{subset_index}块 [{pct_lo:.0f}%~{pct_hi:.0f}%] = {len(train_lines)} 行")
        else:  # first
            train_lines = train_lines[:n]
            print(f"[data] 使用训练子集: 前 {subset_ratio*100:.0f}% = {n} 行")
    result["train"] = train_lines
    print(f"[data] WikiText-103 训练行数: {len(train_lines)}")

    # 验证/测试：单文件
    for split, key in [("validation", "validation"), ("test", "test")]:
        local = os.path.join(DATA_DIR, f"wt103_{split}.parquet")
        if not os.path.exists(local):
            print(f"[data] 警告: 未找到 {local}，跳过 {split}")
            result[key] = []
            continue
        df = pd.read_parquet(local)
        lines = [t for t in df["text"].tolist() if t.strip()]
        result[key] = lines
        print(f"[data] WikiText-103 {split} 行数: {len(lines)}")

    return result


def _build_or_load_tokenizer(dataset: str = "wikitext2"):
    """优先加载已有的 tokenizer，否则用对应数据集训练"""
    if os.path.exists(TOKENIZER_PATH):
        print(f"[data] 加载已存在的分词器: {TOKENIZER_PATH}")
        return load_tokenizer(TOKENIZER_PATH)

    print(f"[data] 未找到分词器，开始训练（用 {dataset} 训练集）...")
    from src.tokenizer import train_tokenizer
    if dataset == "wikitext103":
        texts = _load_wikitext103()["train"]
    else:
        texts = _download_wikitext2()["train"]
    return train_tokenizer(texts, TOKENIZER_PATH)


def _tokenize_and_cache(split_lines: List[str], cache_path: str, tokenizer) -> List[int]:
    """把文本行 tokenize 成 id 列表（逐行拼接，每行之间加 EOS）；若已缓存则直接加载"""
    if os.path.exists(cache_path):
        print(f"[data] 加载缓存: {cache_path}")
        return torch.load(cache_path, weights_only=True).tolist()

    print(f"[data] tokenize {len(split_lines)} 行文本（逐行拼接）...")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    ids = []
    for line in split_lines:
        ids.extend(tokenizer.encode(line).ids)
        ids.append(eos_id)  # 每行之间加 EOS（GPT-2 经典做法；文档级拼接实测无益）

    torch.save(torch.tensor(ids), cache_path)
    print(f"[data] 共 {len(ids)} 个 token，已缓存到 {cache_path}")
    return ids


def get_dataloaders(
    batch_size: int = 8,
    seq_len: int = 1024,
    tokenizer_path: str = TOKENIZER_PATH,
    data_dir: str = DATA_DIR,
    force_retokenize: bool = False,
    include_test: bool = False,
    dataset: str = "wikitext2",
    subset_ratio: float = 1.0,
    subset_mode: str = "first",
    subset_index: int = 0,
    seed: int = 42,
):
    """
    返回 (train_loader, val_loader, tokenizer)；
    若 include_test=True 则额外返回 test_loader。

    参数:
        batch_size:       每个 batch 的序列数
        seq_len:          序列长度（= max_seq_len）
        force_retokenize: 强制重新下载/分词（用于换数据集或重训分词器）
        include_test:     True=同时加载 test split（用于最终评估）
        dataset:          "wikitext2"（默认）或 "wikitext103"
        subset_ratio:     wikitext103 时取多少比例训练数据（1.0=全量）
        subset_mode:      "first"=前N%（默认）| "random"=随机N% | "block"=第N块
        subset_index:     block 模式取第几块（从0开始）
        seed:             random 模式的随机种子
    """
    global TOKENIZER_PATH, DATA_DIR, TRAIN_CACHE, VAL_CACHE
    TOKENIZER_PATH = tokenizer_path
    DATA_DIR = data_dir
    # 数据集 + 子集比例 + 采样方式 用不同的缓存文件，避免互相覆盖
    suffix = f"_{dataset}" if dataset != "wikitext2" else ""
    if subset_ratio < 1.0:
        suffix += f"_{int(subset_ratio*100)}pct"
        if subset_mode == "random":
            suffix += f"_rand{seed}"
        elif subset_mode == "block":
            suffix += f"_blk{subset_index}"
        # first 模式保持旧命名（向后兼容）
    TRAIN_CACHE = os.path.join(data_dir, f"train_ids{suffix}.pt")
    VAL_CACHE = os.path.join(data_dir, f"val_ids{suffix}.pt")
    TEST_CACHE = os.path.join(data_dir, f"test_ids{suffix}.pt")
    # tokenizer 也区分数据集（103 用独立 tokenizer）
    tok_path = TOKENIZER_PATH if dataset == "wikitext2" else \
        os.path.join(data_dir, "tokenizer_wt103.json")
    TOKENIZER_PATH = tok_path

    # 如果强制重建，先清掉缓存
    if force_retokenize:
        for p in (TOKENIZER_PATH, TRAIN_CACHE, VAL_CACHE, TEST_CACHE):
            if os.path.exists(p):
                os.remove(p)
                print(f"[data] 已删除缓存 {p}")

    # 1. 分词器（存在则加载，不存在则训练）
    tokenizer = _build_or_load_tokenizer(dataset)
    vocab_size = tokenizer.get_vocab_size()

    # 2. 文本 → token id（带缓存）
    need_download = (not os.path.exists(TRAIN_CACHE)
                     or not os.path.exists(VAL_CACHE)
                     or (include_test and not os.path.exists(TEST_CACHE)))
    if need_download:
        if dataset == "wikitext103":
            texts = _load_wikitext103(subset_ratio, subset_mode, subset_index, seed)
        else:
            texts = _download_wikitext2()
    else:
        texts = {"train": [], "validation": [], "test": []}

    train_ids = _tokenize_and_cache(texts["train"], TRAIN_CACHE, tokenizer)
    val_ids = _tokenize_and_cache(texts["validation"], VAL_CACHE, tokenizer)

    # 3. 构建 Dataset + DataLoader
    train_ds = TokenDataset(train_ids, seq_len)
    val_ds = TokenDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    print(f"[data] 训练 batch 数: {len(train_loader)}, 验证 batch 数: {len(val_loader)}")
    print(f"[data] 词表大小: {vocab_size}")

    # 可选：test split（最终评估用）
    if include_test:
        test_ids = _tokenize_and_cache(texts["test"], TEST_CACHE, tokenizer)
        test_ds = TokenDataset(test_ids, seq_len)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        print(f"[data] 测试 batch 数: {len(test_loader)}")
        return train_loader, val_loader, test_loader, tokenizer

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # 自测：加载一个小 batch 看看数据长什么样
    train_loader, val_loader, tok = get_dataloaders(
        batch_size=2, seq_len=16, data_dir=DATA_DIR
    )
    x, y = next(iter(train_loader))
    print(f"\nx 形状: {tuple(x.shape)} (batch, seq_len)")
    print(f"y 形状: {tuple(y.shape)} (batch, seq_len)")
    print(f"x[0][:8] 前 8 个 token id: {x[0][:8].tolist()}")
    print(f"y[0][:8] 前 8 个目标 id : {y[0][:8].tolist()}")
    # 验证 y 是 x 右移一位
    assert torch.equal(x[0][1:], y[0][:-1]), "移位错误!"
    print("\n[PASS] 数据加载与移位正确 ✓")
