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
    device: str = "cuda",
):
    """用 KV-cache 逐 token 生成文本，返回完整文本"""
    model.eval()

    # 1. 编码 prompt
    prompt_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print(f"prompt: {prompt!r} → {len(prompt_ids)} tokens")

    generated = list(prompt_ids)

    # 2. 逐 token 生成（KV-cache 加速）
    past_kvs = None
    for _ in range(max_new_tokens):
        logits, past_kvs = model(x, past_kvs)          # 每步只输入 1 个新 token
        logits = logits[:, -1, :] / temperature        # 最后位置，除以温度控制多样性

        # 3. 采样策略
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/final.pt", help="模型权重路径")
    p.add_argument("--prompt", type=str, default="The meaning of life is", help="生成起始文本")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--top_k", type=int, default=None, help="top-k 采样；None 表示 greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载分词器 + 模型
    tokenizer = load_tokenizer(TOKENIZER_PATH)
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
        device=device,
    )
    print("\n" + "=" * 60)
    print("生成结果:")
    print("=" * 60)
    print(text)
    print("=" * 60)
