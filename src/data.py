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


def _build_or_load_tokenizer():
    """优先加载已有的 tokenizer.json，否则用 WikiText-2 训练"""
    if os.path.exists(TOKENIZER_PATH):
        print(f"[data] 加载已存在的分词器: {TOKENIZER_PATH}")
        return load_tokenizer(TOKENIZER_PATH)

    print("[data] 未找到分词器，开始训练（用 WikiText-2 训练集）...")
    from src.tokenizer import train_tokenizer
    texts = _download_wikitext2()["train"]
    return train_tokenizer(texts, TOKENIZER_PATH)


def _group_into_docs(lines: List[str], eos_id: int, tokenizer) -> List[int]:
    """
    文档级拼接：把连续的文本行按空行分成"文档"，
    同一文档内的行串成一个连续序列（不加 EOS），只在文档边界加 EOS。

    对比旧的"逐行加 EOS"方式，好处是块内上下文连贯，
    减少"每行开头无前文可依赖"导致的无效预测。

    返回: 拼好的 token id 列表
    """
    ids = []
    current_doc = []  # 当前文档的 token

    for line in lines:
        if line.strip():
            current_doc.extend(tokenizer.encode(line.strip()).ids)
        else:
            # 空行 = 段落分隔。如果当前文档非空，则在此收尾（加 EOS）
            if current_doc:
                current_doc.append(eos_id)
                ids.extend(current_doc)
                current_doc = []

    # 处理最后一段没有空行结尾的文档
    if current_doc:
        current_doc.append(eos_id)
        ids.extend(current_doc)

    return ids


def _tokenize_and_cache(split_lines: List[str], cache_path: str, tokenizer,
                        use_doc_concat: bool = False) -> List[int]:
    """把文本行 tokenize 成 id 列表；若已缓存则直接加载"""
    if os.path.exists(cache_path):
        print(f"[data] 加载缓存: {cache_path}")
        return torch.load(cache_path, weights_only=True).tolist()

    mode = "文档级拼接" if use_doc_concat else "逐行拼接"
    print(f"[data] tokenize {len(split_lines)} 行文本（{mode}）...")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    if use_doc_concat:
        ids = _group_into_docs(split_lines, eos_id, tokenizer)
    else:
        # 逐行拼接：每行之间加 EOS（GPT-2 经典做法，实测在本数据集上更优）
        ids = []
        for line in split_lines:
            ids.extend(tokenizer.encode(line).ids)
            ids.append(eos_id)

    torch.save(torch.tensor(ids), cache_path)
    print(f"[data] 共 {len(ids)} 个 token，已缓存到 {cache_path}")
    return ids


def get_dataloaders(
    batch_size: int = 8,
    seq_len: int = 1024,
    tokenizer_path: str = TOKENIZER_PATH,
    data_dir: str = DATA_DIR,
    force_retokenize: bool = False,
    use_doc_concat: bool = False,
    include_test: bool = False,
):
    """
    返回 (train_loader, val_loader, tokenizer)；
    若 include_test=True 则额外返回 test_loader。

    参数:
        batch_size:       每个 batch 的序列数
        seq_len:          序列长度（= max_seq_len）
        force_retokenize: 强制重新下载/分词（用于换数据集或重训分词器）
        use_doc_concat:   True=文档级拼接；False=逐行拼接（默认，实测更优）
        include_test:     True=同时加载 test split（用于最终评估）
    """
    global TOKENIZER_PATH, DATA_DIR, TRAIN_CACHE, VAL_CACHE
    TOKENIZER_PATH = tokenizer_path
    DATA_DIR = data_dir
    # 两种模式用不同的缓存文件，避免互相覆盖
    suffix = "_doc" if use_doc_concat else ""
    TRAIN_CACHE = os.path.join(data_dir, f"train_ids{suffix}.pt")
    VAL_CACHE = os.path.join(data_dir, f"val_ids{suffix}.pt")
    TEST_CACHE = os.path.join(data_dir, f"test_ids{suffix}.pt")

    # 如果强制重建，先清掉缓存
    if force_retokenize:
        for p in (TOKENIZER_PATH, TRAIN_CACHE, VAL_CACHE, TEST_CACHE):
            if os.path.exists(p):
                os.remove(p)
                print(f"[data] 已删除缓存 {p}")

    # 1. 分词器（存在则加载，不存在则训练）
    tokenizer = _build_or_load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # 2. 文本 → token id（带缓存）
    need_download = (not os.path.exists(TRAIN_CACHE)
                     or not os.path.exists(VAL_CACHE)
                     or (include_test and not os.path.exists(TEST_CACHE)))
    if need_download:
        texts = _download_wikitext2()
    else:
        texts = {"train": [], "validation": [], "test": []}

    train_ids = _tokenize_and_cache(texts["train"], TRAIN_CACHE, tokenizer, use_doc_concat)
    val_ids = _tokenize_and_cache(texts["validation"], VAL_CACHE, tokenizer, use_doc_concat)

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
        test_ids = _tokenize_and_cache(texts["test"], TEST_CACHE, tokenizer, use_doc_concat)
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
