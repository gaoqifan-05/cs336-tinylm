"""
文本生成脚本 —— 加载训练好的模型，用 KV-cache 逐 token 生成文本

支持两种采样策略：
- greedy：每次都选概率最高的 token（结果稳定但单调）
- top-k：只从概率最高的 k 个 token 里随机采样（结果多样，更自然）

运行：
    python src/generate.py --prompt "The meaning of life is"
    python src/generate.py --prompt "Once upon a time" --top_k 50 --max_new_tokens 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model import DecoderOnlyLM, ModelConfig
from src.data import TOKENIZER_PATH
from src.tokenizer import load_tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    top_k: int = None,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
):
    """用 KV-cache 逐 token 生成文本，返回完整文本"""
    model.eval()

    # 1. 编码 prompt
    prompt_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print(f"prompt: {prompt!r} → {len(prompt_ids)} tokens")

    # 保护：生成总长度不能超过 max_seq_len（RoPE 表长度）
    max_seq_len = model.cfg.max_seq_len
    max_new_tokens = min(max_new_tokens, max_seq_len - len(prompt_ids))
    if max_new_tokens <= 0:
        raise ValueError(f"prompt 长度 {len(prompt_ids)} 已超过 max_seq_len={max_seq_len}")
    print(f"将生成 {max_new_tokens} 个 token（上限受 max_seq_len 限制）")

    generated = list(prompt_ids)

    # 2. 逐 token 生成（KV-cache 加速）
    past_kvs = None
    for _ in range(max_new_tokens):
        logits, past_kvs = model(x, past_kvs)          # 每步只输入 1 个新 token
        logits = logits[:, -1, :] / temperature        # 最后位置，除以温度控制多样性

        # 3. repetition penalty：对已生成过的 token 打折扣，抑制重复循环
        if repetition_penalty != 1.0:
            for pid in generated:
                if logits[0, pid] > 0:
                    logits[0, pid] /= repetition_penalty
                else:
                    logits[0, pid] *= repetition_penalty

        # 4. 采样策略
        if top_k is not None:
            # top-k：只保留概率最高的 k 个
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)   # 按概率采样
        next_id = next_tok.item()

        # 遇到结束符就停
        if next_id == tokenizer.token_to_id("<|endoftext|>"):
            break

        generated.append(next_id)
        x = next_tok

    # 4. 解码回文本
    text = tokenizer.decode(generated)
    return text


def find_latest_ckpt(ckpt_dir: str = "checkpoints") -> str:
    """自动找到最新的实验目录里的最佳模型（优先 best.pt，否则 final.pt）"""
    if not os.path.isdir(ckpt_dir):
        return os.path.join(ckpt_dir, "final.pt")

    # 找出所有 run_* 目录，按名字（含时间戳）倒序取最新
    runs = [d for d in os.listdir(ckpt_dir) if d.startswith("run_")]
    if not runs:
        # 兼容旧的直接放 final.pt 的情况
        old = os.path.join(ckpt_dir, "final.pt")
        return old if os.path.exists(old) else os.path.join(ckpt_dir, "final.pt")

    runs.sort(reverse=True)   # 名字含 YYYYMMDD_HHMMSS，字典序=时间序

    # 最新 run 里优先取 best.pt（验证集最优），否则用 final.pt
    latest_dir = os.path.join(ckpt_dir, runs[0])
    best = os.path.join(latest_dir, "best.pt")
    if os.path.exists(best):
        return best
    return os.path.join(latest_dir, "final.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="模型权重路径；默认自动选最新的实验目录")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="tokenizer json 路径；默认用 data/tokenizer.json（wikitext2）。\n"
                        "wt103 模型需指定 data/tokenizer_wt103.json")
    p.add_argument("--prompt", type=str, default="The meaning of life is", help="生成起始文本")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--top_k", type=int, default=None, help="top-k 采样；None 表示 greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="重复惩罚（>1 抑制重复，如 1.2）；1.0 表示关闭")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 自动定位最新 checkpoint
    if args.ckpt is None:
        args.ckpt = find_latest_ckpt()

    # 加载分词器 + 模型
    tok_path = args.tokenizer or TOKENIZER_PATH
    tokenizer = load_tokenizer(tok_path)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DecoderOnlyLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载模型: {args.ckpt} ({model.param_count()/1e6:.2f}M)")

    # 生成
    text = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print("\n" + "=" * 60)
    print("生成结果:")
    print("=" * 60)
    print(text)
    print("=" * 60)
