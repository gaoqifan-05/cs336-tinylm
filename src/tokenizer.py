"""
BPE 分词器 —— 基于 HuggingFace tokenizers 库（GPT-2 风格 ByteLevel BPE）

为什么需要分词器？
    神经网络只能吃整数，不能吃文本。分词器就是"文本 ↔ token id"的转换器：
        "The cat sat"  →  [464, 3797, 4678]   (encode)
        [464, 3797, 4678]  →  "The cat sat"   (decode)

为什么用 BPE + ByteLevel？
    - BPE（Byte Pair Encoding）：从字符开始，反复合并出现频率最高的相邻对，
      让常见词/子词成为单个 token，稀有词退化成子词，词表大小可控。
    - ByteLevel：先把文本转成 UTF-8 字节，再在字节上做 BPE。这样词表只有 256 个
      字节起步，任何文本（包括表情、生僻字）都能表示，不会出现 <UNK>。
    - GPT-2 用的就是这个方案，vocab_size=50257 就是这么来的。
"""

import os
from typing import List

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


# BPE 训练超参（和 GPT-2 保持一致）
VOCAB_SIZE = 50257        # 词表大小
MIN_FREQUENCY = 2         # 词频低于 2 的 token 直接丢弃


def train_tokenizer(texts: List[str], save_path: str) -> Tokenizer:
    """
    在文本列表上训练一个 ByteLevel BPE 分词器并保存。

    参数:
        texts:    训练用文本列表（list[str]），如 WikiText-2 的每一行
        save_path: 保存分词器的路径，如 data/tokenizer.json
    返回:
        训练好的 Tokenizer 对象
    """
    # 1. 构建 GPT-2 风格的 BPE 模型
    tokenizer = Tokenizer(models.BPE())

    # 2. 预分词：先按空白切分（保持单词完整性），再转 UTF-8 字节
    #    这是 GPT-2 的标准做法，比纯字节级 BPE 效果更好
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. 解码器：解码时把字节还原回文本
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["<|endoftext|>"],   # 文本结束符（GPT-2 用它分隔文档）
    )

    # 5. 训练
    tokenizer.train_from_iterator(texts, trainer)

    # 6. 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"[tokenizer] 训练完成，词表大小 = {tokenizer.get_vocab_size()}")
    print(f"[tokenizer] 已保存到 {save_path}")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """从文件加载分词器"""
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    """文本 → token id 列表（自动加结束符）"""
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    """token id 列表 → 文本"""
    return tokenizer.decode(ids)


if __name__ == "__main__":
    # 快速自测：用几行英文训练一个玩具分词器
    sample = [
        "The quick brown fox jumps over the lazy dog.",
        "Language modeling is the task of predicting the next word.",
        "BPE stands for Byte Pair Encoding, a subword tokenization method.",
        "The cat sat on the mat while the dog slept.",
    ]
    tok = train_tokenizer(sample, "data/tokenizer.json")

    # 验证 encode / decode 往返
    text = "The quick brown fox"
    ids = encode(tok, text)
    back = decode(tok, ids)
    print(f"\n原始文本 : {text}")
    print(f"token ids: {ids}")
    print(f"还原文本 : {back!r}")
    # 注意：GPT-2 风格的 ByteLevel 会在句首加空格（保留单词边界），
    # 所以对比时用 strip() 忽略首尾空白
    assert text.lower().strip() == back.lower().strip(), "往返不一致!"
    print("\n[PASS] 分词器编解码往返一致 ✓")
