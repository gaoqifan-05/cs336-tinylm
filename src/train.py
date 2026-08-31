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
    p.add_argument("--dropout", type=float, default=0.1, help="模型 dropout 概率")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="标签平滑系数；0 表示关闭（0.1 是常用值）")
    p.add_argument("--warmup_steps", type=int, default=200, help="warmup 步数")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="梯度裁剪范数（None/0 表示不裁剪）")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA 衰减系数；0 表示关闭 EMA")
    p.add_argument("--ema_start_step", type=int, default=0,
                   help="EMA 延迟启用的步数；0 表示训练一开始就启用（默认）")
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
    p.add_argument("--tag", type=str, default=None,
                   help="实验标签，会拼进文件名（如 seq512_3ep），便于区分不同实验")
    p.add_argument("--resume", type=str, default=None,
                   help="从指定 checkpoint 热启动（加载模型权重继续训练）")
    p.add_argument("--fine_tune", type=float, default=0.0,
                   help="精修模式：用此固定小 LR 训练（如 1e-5），不用 warmup/cosine")
    p.add_argument("--tb", action="store_true",
                   help="启用 TensorBoard 实时监控（logs 写到 outputs/runs/）")
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


class EMA:
    """
    指数移动平均（Exponential Moving Average）。
    维护一份模型权重的滑动平均，训练后期参数在最优解附近震荡时，
    EMA 权重相当于"平均解"，更平滑、泛化更好。用 EMA 权重做验证/生成。

    更新公式：ema = decay * ema + (1 - decay) * param
    用法：
        ema = EMA(model, decay=0.999)
        每步训练后: ema.update(model)
        验证时:     ema.apply_shadow()  # 临时换到 EMA 权重
                    evaluate(...)
                    ema.restore()       # 换回在线权重
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}                      # 保存 EMA 权重
        self.backup = {}                      # 保存在线权重（临时用）
        self.initialized = False              # 是否已初始化（延迟启用用）
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        """每步训练后更新 EMA 权重"""
        if self.decay <= 0:
            return
        if not self.initialized:
            return  # 尚未到启用步数，不更新
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def activate(self, model):
        """
        延迟启用：把 shadow 重置为当前权重。
        这样 EMA 的起点就是"已成熟的在线权重"，几乎零滞后，
        从启用时刻开始平滑后续训练。
        """
        if self.decay <= 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].copy_(param.data)
        self.initialized = True

    @torch.no_grad()
    def apply_shadow(self, model):
        """把模型权重临时换成 EMA 权重（用于验证/生成）"""
        if self.decay <= 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        """恢复在线权重"""
        if self.decay <= 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


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
        dropout=args.dropout,
    )
    model = DecoderOnlyLM(cfg).to(device)
    print(f"模型参数量: {model.param_count()/1e6:.2f}M")

    # 可选：从 checkpoint 热启动（加载模型权重继续训练）
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        print(f"已从 {args.resume} 加载权重继续训练（原验证 PPL ≈ {resume_ckpt.get('val_ppl', '?')}）")

    # 3. 优化器（AdamW + weight decay）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 4. LR scheduler
    if args.fine_tune > 0:
        # 精修模式：固定极小 LR，不用 warmup/cosine，避免打乱已收敛权重
        for g in optimizer.param_groups:
            g["lr"] = args.fine_tune
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        print(f"精修模式：固定 LR = {args.fine_tune}")
    else:
        # 正常模式：Cosine LR（带 warmup）
        total_steps = args.max_steps or (len(train_loader) * args.epochs)
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

    # 6. EMA 指数移动平均（decay=0 则关闭）
    ema = EMA(model, decay=args.ema_decay)
    if args.ema_decay > 0:
        print(f"EMA 已开启，decay = {args.ema_decay}")

    # 记录曲线
    train_losses, val_ppls, lrs = [], [], []

    # 时间戳：每次训练一个独立目录，避免覆盖之前的实验
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{args.tag}_" if args.tag else ""
    run_name = f"run_{timestamp}_{tag}{args.seq_len}seq_{args.epochs}ep"
    run_out_dir = os.path.join(args.out_dir, run_name)
    run_ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(run_out_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    print(f"实验目录: {run_name}")

    # 可选：TensorBoard 实时监控
    tb_writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = os.path.join(args.out_dir, "runs", run_name)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard 日志: {tb_log_dir}")

    print("\n开始训练...")
    global_step = 0
    best_ppl = float("inf")   # 最佳验证 PPL（用于保存 best.pt）
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
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    label_smoothing=args.label_smoothing,
                )

            # 反向（scaler 处理 fp16 梯度）
            scaler.scale(loss).backward()
            # 梯度裁剪（在 AMP 下需先 unscale 还原梯度再裁剪，防止被裁剪到错误尺度）
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA 延迟启用：到达启用步数时，用当前权重初始化 shadow
            if (args.ema_decay > 0 and not ema.initialized
                    and global_step >= args.ema_start_step):
                ema.activate(model)
                if args.ema_start_step > 0:
                    print(f"  → EMA 在 step {global_step} 启用（从当前权重初始化）")

            # 每步训练后更新 EMA 权重（未启用时 update 内部直接跳过）
            ema.update(model)

            train_losses.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])

            # TensorBoard 记录（loss / lr 每步）
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss.item(), global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if global_step % args.log_every == 0:
                print(f"[epoch {epoch+1}/{args.epochs}] step {global_step} | "
                      f"loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            # 验证（EMA 已启用才用 EMA 权重；未启用用在线权重）
            if global_step % args.eval_every == 0 and global_step > 0:
                if ema.initialized:
                    ema.apply_shadow(model)
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                    ema.restore(model)
                else:
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                val_ppls.append((global_step, val_ppl))
                print(f"  → 验证 loss {val_loss:.4f} | PPL {val_ppl:.2f}")

                # TensorBoard 记录验证 PPL / loss
                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss", val_loss, global_step)
                    tb_writer.add_scalar("val/ppl", val_ppl, global_step)

                # 保存检查点（带步数，多个检查点共存）
                ckpt_path = os.path.join(run_ckpt_dir, f"step_{global_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "step": global_step,
                    "val_ppl": val_ppl,
                    "train_loss": loss.item(),
                }, ckpt_path)

                # 若验证 PPL 比历史最优更好，额外保存 best.pt（保存 EMA 权重）
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    ema.apply_shadow(model)
                    best_path = os.path.join(run_ckpt_dir, "best.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "config": cfg,
                        "step": global_step,
                        "val_ppl": val_ppl,
                        "train_loss": loss.item(),
                        "ema": True,
                    }, best_path)
                    ema.restore(model)
                    print(f"    ★ 新最佳 PPL {val_ppl:.2f}，已保存 {best_path}")

            global_step += 1

    # 关闭 TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()

    # 6. 保存最终模型 + 曲线数据（用 EMA 权重，泛化更好）
    ema.apply_shadow(model)
    final_path = os.path.join(run_ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "step": global_step,
        "val_ppl": val_ppls[-1][1] if val_ppls else None,
        "ema": True,
    }, final_path)
    ema.restore(model)
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
    fig_path = os.path.join(run_out_dir, "training_curves.png")
    fig.savefig(fig_path, dpi=150)
    print(f"曲线图已保存: {fig_path}")

    # 8. 打印总结（写进 README / CV 用）
    if val_ppls:
        print("\n========== 训练总结 ==========")
        print(f"模型参数量: {model.param_count()/1e6:.2f}M")
        print(f"最终验证集 Perplexity: {val_ppls[-1][1]:.2f}")
        print(f"最佳验证集 Perplexity: {best_ppl if best_ppl != float('inf') else 'N/A'}（见 best.pt）")
        print(f"（随机初始化时 PPL ≈ {vocab_size}，训练后明显下降说明模型学到语言规律）")


if __name__ == "__main__":
    main()
