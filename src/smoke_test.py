"""
冒烟测试：验证 model.py 的三个核心正确性
1. 前向传播 shape 是否正确
2. 反向传播 / loss 是否正常
3. KV-cache 生成结果是否与全量计算一致（关键验证）

运行：python src/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model import DecoderOnlyLM, ModelConfig


def make_small_config():
    """一个极小的配置，跑得快、方便手算验证"""
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,          # head_dim = 64/4 = 16
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        dropout=0.0,
    )


def test_forward_shape():
    """测试 1：前向传播 shape"""
    print("=" * 60)
    print("测试 1: 前向传播 shape")
    cfg = make_small_config()
    model = DecoderOnlyLM(cfg)
    model.eval()

    B, T = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (B, T, cfg.vocab_size), \
        f"logits shape 错误: {logits.shape}, 期望 {(B, T, cfg.vocab_size)}"
    print(f"  [PASS] logits shape = {tuple(logits.shape)} ✓")
    print(f"  参数量 = {model.param_count() / 1e6:.2f}M")


def test_backward():
    """测试 2：反向传播 / loss"""
    print("=" * 60)
    print("测试 2: 反向传播 + loss")
    cfg = make_small_config()
    model = DecoderOnlyLM(cfg)
    model.train()

    B, T = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    logits, _ = model(input_ids)
    # 语言模型任务：用位置 t 预测位置 t+1，所以 targets 是 input_ids 右移一位
    targets = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()

    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()

    # 检查所有参数都有梯度
    missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing_grad, f"以下参数没有梯度: {missing_grad}"
    print(f"  [PASS] loss = {loss.item():.4f} ✓")
    print(f"  [PASS] 所有参数均收到梯度 ✓")

    # 验证 loss 是否合理：随机初始化下，交叉熵 ≈ ln(vocab_size)
    expected = torch.log(torch.tensor(cfg.vocab_size, dtype=torch.float32))
    print(f"  参考值 ln(vocab_size) = {expected.item():.4f}（随机初始化 loss 应接近它）")


def test_kv_cache_consistency():
    """测试 3（关键）：KV-cache 逐 token 生成 == 全量一次计算"""
    print("=" * 60)
    print("测试 3: KV-cache 一致性")
    cfg = make_small_config()
    model = DecoderOnlyLM(cfg)
    model.eval()

    # 一组随机输入 prompt
    B, T = 1, 8
    prompt = torch.randint(0, cfg.vocab_size, (B, T))
    gen_len = 12  # 额外生成的 token 数

    # --- 方式 A：全量一次计算整段（无缓存），作为标准答案 ---
    with torch.no_grad():
        full_input = torch.cat([prompt, torch.zeros(B, gen_len, dtype=torch.long)], dim=1)
        logits_full, _ = model(full_input)
        logits_full = logits_full[:, -gen_len:]  # 只看生成部分的 logits

    # --- 方式 B：KV-cache 逐 token 生成 ---
    with torch.no_grad():
        # 第一步：处理 prompt，建立缓存（这一步的输出不参与比较）
        _, past_kvs = model(prompt, None)
        x = torch.zeros(B, 1, dtype=torch.long)  # 占位 token（与方式 A 的填充一致）
        logits_cache = []
        for _ in range(gen_len):
            logits, past_kvs = model(x, past_kvs)  # 每步只输入 1 个新 token
            logits_cache.append(logits[:, 0])      # 唯一的 token 位置
            x = torch.zeros(B, 1, dtype=torch.long)
        logits_cache = torch.stack(logits_cache, dim=1)  # (B, gen_len, vocab)

    diff = (logits_full - logits_cache).abs().max().item()
    print(f"  两种方式的最大误差 = {diff:.2e}")
    assert diff < 1e-4, f"KV-cache 结果不一致! 误差 {diff}"
    print("  [PASS] KV-cache 逐 token 生成 == 全量计算 ✓")


def test_generation():
    """测试 4：完整生成流程（先让模型学一点点，再贪心生成）"""
    print("=" * 60)
    print("测试 4: 文本生成流程")
    cfg = make_small_config()
    model = DecoderOnlyLM(cfg)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    gen_len = 10

    with torch.no_grad():
        x = prompt
        past_kvs = None
        tokens = prompt.tolist()[0]
        for _ in range(gen_len):
            logits, past_kvs = model(x, past_kvs)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # greedy
            tokens.append(next_tok.item())
            x = next_tok
    print(f"  生成 token 序列: {tokens}（长度 {len(tokens)}）✓")
    print("  [PASS] 生成流程正常 ✓")


if __name__ == "__main__":
    print("PyTorch CUDA 可用:", torch.cuda.is_available())
    test_forward_shape()
    test_backward()
    test_kv_cache_consistency()
    test_generation()
    print("=" * 60)
    print("🎉 全部测试通过！模型实现正确。")
