"""
SFT 微调脚本 —— Supervised Fine-Tuning

原理：
- 在预训练好的 best_model.pt 基础上，用"问题-答案"指令数据继续训练
- 与预训练的唯一本质区别：loss masking（只对 Answer 部分算 loss）
  - Question 部分是"输入"，不该被预测 → loss 权重 0
  - Answer 部分是"要学的内容" → loss 权重 1

运行：
    python src/sft.py                        # 默认微调，存到 checkpoints/sft/
    python src/sft.py --epochs 5 --lr 1e-4
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model import DecoderOnlyLM
from src.data import TOKENIZER_PATH
from src.tokenizer import load_tokenizer

# 预训练好的起点模型（wt103 续跑后的最优模型，PPL 89.87）
PRETRAINED = "checkpoints/run_20260901_110553_wt103_blk1_d1024_bs12_cont_256seq_1ep/best.pt"
TOKENIZER_WT103 = "data/tokenizer_wt103.json"
SFT_DATA_PATH = "data/sft_data_alpaca10k.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=PRETRAINED, help="预训练模型路径")
    p.add_argument("--tokenizer", type=str, default=TOKENIZER_WT103,
                   help="tokenizer json 路径（wt103 模型需用 tokenizer_wt103.json）")
    p.add_argument("--data", type=str, default=SFT_DATA_PATH, help="SFT 指令数据 JSON 路径")
    p.add_argument("--epochs", type=int, default=3, help="微调轮数")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="微调学习率（小，避免破坏预训练知识）")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=200,
                   help="每多少 batch 打印一次进度")
    p.add_argument("--out_dir", type=str, default="checkpoints/sft")
    return p.parse_args()


def load_sft_data(path):
    """加载 SFT 数据，格式化为 (input_ids, mask) 列表"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def tokenize_example(tokenizer, ex):
    """
    把一条问答格式化为 token 序列 + answer 掩码。

    返回:
        ids:  完整序列的 token id 列表
        mask: 与 ids 等长，1 表示"该位置要算 loss"（即 Answer 部分），0 表示忽略
    """
    # 格式化："Question: ...\nAnswer: ..."
    # Alpaca 的 input 字段作为额外上下文拼进 Question
    question = f"Question: {ex['instruction']}"
    if ex.get("input"):
        question += f"\nContext: {ex['input']}"
    question += "\nAnswer:"
    answer = f" {ex['response']}"

    q_ids = tokenizer.encode(question).ids
    a_ids = tokenizer.encode(answer).ids

    ids = q_ids + a_ids
    # Answer 部分（含答案 token + 结束符）都要算 loss
    mask = [0] * len(q_ids) + [1] * len(a_ids)
    return ids, mask


def collate(batch, tokenizer, seq_len):
    """
    把一个 batch 的 (ids, mask) 填充到等长，并转成张量。
    返回 x, y, loss_mask
    """
    pad_id = tokenizer.token_to_id("<|endoftext|>")

    # 截断到 seq_len
    all_ids = [ids[:seq_len] for ids, _ in batch]
    all_mask = [mask[:seq_len] for _, mask in batch]

    max_len = max(len(ids) for ids in all_ids)
    max_len = min(max_len, seq_len)

    padded_ids, padded_mask = [], []
    for ids, mask in zip(all_ids, all_mask):
        pad = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad)
        padded_mask.append(mask + [0] * pad)  # padding 位置不算 loss

    x = torch.tensor(padded_ids, dtype=torch.long)
    y = torch.tensor(padded_ids, dtype=torch.long)
    loss_mask = torch.tensor(padded_mask, dtype=torch.float)

    return x, y, loss_mask


def train_step(model, x, y, loss_mask, optimizer, scaler, args, device):
    """单个训练步（返回 loss）"""
    model.train()
    x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        logits, _ = model(x)  # (B, T, vocab)
        # 用位置 t 预测 t+1
        logits = logits[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()
        mask = loss_mask[:, 1:].contiguous()

        # 交叉熵（reduction='none' 才能乘 mask）
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss * mask.reshape(-1)
        loss = loss.sum() / (mask.sum() + 1e-8)  # 只对 Answer 部分平均

    scaler.scale(loss).backward()
    if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 加载预训练模型
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DecoderOnlyLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载预训练模型: {args.pretrained} ({model.param_count()/1e6:.2f}M, PPL {ckpt.get('val_ppl', '?')})")

    # 2. 加载 tokenizer + SFT 数据
    tokenizer = load_tokenizer(args.tokenizer)
    data = load_sft_data(args.data)
    print(f"加载 {len(data)} 条 SFT 指令")

    # 3. tokenize（每条 → ids + answer mask）
    examples = [tokenize_example(tokenizer, ex) for ex in data]
    avg_len = sum(len(ids) for ids, _ in examples) / len(examples)
    print(f"平均序列长度: {avg_len:.0f} tokens")

    # 4. 优化器（小 lr，AdamW）
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # 5. 训练循环
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n开始 SFT 微调（{args.epochs} epochs, lr={args.lr}）...")
    n_batches = math.ceil(len(examples) / args.batch_size)

    for epoch in range(args.epochs):
        # 打乱数据
        indices = torch.randperm(len(examples))
        epoch_loss = 0.0
        t0 = time.time()
        for i in range(n_batches):
            batch_idx = indices[i * args.batch_size: (i + 1) * args.batch_size]
            batch = [examples[j] for j in batch_idx]
            x, y, mask = collate(batch, tokenizer, args.seq_len)
            loss = train_step(model, x, y, mask, optimizer, scaler, args, device)
            epoch_loss += loss

            # 进度显示：每 log_every 个 batch 打印一次
            if (i + 1) % args.log_every == 0:
                avg_so_far = epoch_loss / (i + 1)
                elapsed = time.time() - t0
                pct = (i + 1) / n_batches * 100
                speed = (i + 1) / elapsed
                remain = (n_batches - i - 1) / speed
                print(f"[epoch {epoch+1}/{args.epochs}] "
                      f"batch {i+1}/{n_batches} ({pct:.0f}%) | "
                      f"loss {avg_so_far:.4f} | "
                      f"{speed:.1f} batch/s | "
                      f"已用 {elapsed:.0f}s 剩余~{remain:.0f}s", flush=True)

        avg = epoch_loss / n_batches
        print(f"[epoch {epoch+1}/{args.epochs}] 完成 | 平均 loss {avg:.4f} | "
              f"用时 {time.time()-t0:.0f}s", flush=True)

    # 6. 保存微调后的模型
    out_path = os.path.join(run_dir, "sft_model.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "pretrained_ppl": ckpt.get("val_ppl"),
        "sft_epochs": args.epochs,
        "sft_lr": args.lr,
    }, out_path)
    print(f"\nSFT 模型已保存: {out_path}")


if __name__ == "__main__":
    main()
