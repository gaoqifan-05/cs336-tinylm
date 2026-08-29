"""
评估脚本 —— 在验证集 / 测试集上计算模型的 loss 和 perplexity（学术规范：用 test split 报告最终数字）

运行：
    python src/evaluate.py                          # 自动加载最新 best.pt，在 test 上评估
    python src/evaluate.py --ckpt path/to/best.pt   # 指定模型
    python src/evaluate.py --split validation       # 在验证集上评估
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model import DecoderOnlyLM
from src.data import get_dataloaders
from src.generate import find_latest_ckpt


@torch.no_grad()
def evaluate_split(model, loader, device):
    """在指定 DataLoader 上计算平均 loss 和 PPL"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="模型权重路径；默认自动选最新 best.pt")
    p.add_argument("--split", type=str, default="test",
                   choices=["test", "validation"], help="在哪个 split 上评估")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        args.ckpt = find_latest_ckpt()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = DecoderOnlyLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载: {args.ckpt} ({model.param_count()/1e6:.2f}M)")

    # 加载数据（test split 需要 include_test=True 才返回）
    result = get_dataloaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        include_test=(args.split == "test"),
    )
    if args.split == "test":
        train_loader, val_loader, test_loader, tokenizer = result
        loader = test_loader
    else:
        train_loader, val_loader, tokenizer = result
        loader = val_loader

    avg_loss, ppl = evaluate_split(model, loader, device)
    print(f"\n========== 评估结果 ==========")
    print(f"split: {args.split}")
    print(f"平均 loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    print(f"（随机初始化时 PPL ≈ {tokenizer.get_vocab_size()}）")
