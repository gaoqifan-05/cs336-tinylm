"""
预训练脚本 —— 用 WikiText-2 训练 Decoder-Only 小模型

训练技术（CS336 标配，也是 CV 亮点）：
- 混合精度 AMP（torch.amp）：fp16 计算加速 + fp32 主权重保精度
- Cosine LR scheduler：学习率先线性 warmup，再按余弦曲线衰减
- Weight decay：AdamW 的 L2 正则化，抑制过拟合

运行：
    python src/train.py                      # 用默认配置训练
    python src/train.py --epochs 3 --batch_size 16
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model import DecoderOnlyLM, ModelConfig
from src.data import get_dataloaders


# 默认超参（几十分钟可跑完的规模）
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=8, help="每个 batch 的序列数")
    p.add_argument("--seq_len", type=int, default=256, help="序列长度（训练用短些，快）")
    p.add_argument("--lr", type=float, default=3e-4, help="峰值学习率")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--warmup_steps", type=int, default=200, help="warmup 步数")
    p.add_argument("--eval_every", type=int, default=500, help="每多少步验证一次")
    p.add_argument("--log_every", type=int, default=50, help="每多少步打印一次")
    p.add_argument("--max_steps", type=int, default=None, help="最大步数（调试用）")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """在验证集上计算平均 loss 和 perplexity"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_loader:
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
    model.train()
    return avg_loss, ppl


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 数据
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size, seq_len=args.seq_len
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"词表大小: {vocab_size}")

    # 2. 模型
    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=0.1,
    )
    model = DecoderOnlyLM(cfg).to(device)
    print(f"模型参数量: {model.param_count()/1e6:.2f}M")

    # 3. 优化器（AdamW + weight decay）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 4. Cosine LR scheduler（带 warmup）
    #    总步数：先算个估计值，保证 schedule 完整
    total_steps = args.max_steps or (len(train_loader) * args.epochs)
    # warmup 占比不能超过 0.3（否则 pct_start 超过 1 报错）
    warmup_pct = min(args.warmup_steps / total_steps, 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_pct,
        anneal_strategy="cos",
    )

    # 5. AMP 混合精度（自动缩放梯度，防止 fp16 下溢）
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # 记录曲线
    train_losses, val_ppls, lrs = [], [], []
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("\n开始训练...")
    global_step = 0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            if args.max_steps and global_step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)
            model.train()

            # 前向（AMP：自动 cast 到 fp16 计算）
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )

            # 反向（scaler 处理 fp16 梯度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])

            if global_step % args.log_every == 0:
                print(f"[epoch {epoch+1}/{args.epochs}] step {global_step} | "
                      f"loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            # 验证
            if global_step % args.eval_every == 0 and global_step > 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                val_ppls.append((global_step, val_ppl))
                print(f"  → 验证 loss {val_loss:.4f} | PPL {val_ppl:.2f}")

                # 保存检查点
                ckpt_path = os.path.join(args.ckpt_dir, f"step_{global_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "step": global_step,
                    "val_ppl": val_ppl,
                    "train_loss": loss.item(),
                }, ckpt_path)

            global_step += 1

    # 6. 保存最终模型 + 曲线数据
    final_path = os.path.join(args.ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "step": global_step,
        "val_ppl": val_ppls[-1][1] if val_ppls else None,
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")

    # 7. 画 loss / lr / ppl 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(train_losses)
    ax[0].set_title("Training Loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[1].plot(lrs)
    ax[1].set_title("Learning Rate (Cosine)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("lr")
    if val_ppls:
        steps = [s for s, _ in val_ppls]
        ppls = [p for _, p in val_ppls]
        ax[2].plot(steps, ppls, marker="o")
        ax[2].set_title("Validation Perplexity"); ax[2].set_xlabel("step"); ax[2].set_ylabel("PPL")
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "training_curves.png")
    fig.savefig(fig_path, dpi=150)
    print(f"曲线图已保存: {fig_path}")

    # 8. 打印总结（写进 README / CV 用）
    if val_ppls:
        print("\n========== 训练总结 ==========")
        print(f"模型参数量: {model.param_count()/1e6:.2f}M")
        print(f"最终验证集 Perplexity: {val_ppls[-1][1]:.2f}")
        print(f"（随机初始化时 PPL ≈ {vocab_size}，训练后明显下降说明模型学到语言规律）")


if __name__ == "__main__":
    main()
