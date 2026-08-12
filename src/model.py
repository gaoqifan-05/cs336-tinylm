"""
Decoder-Only Transformer 语言模型 —— 从零实现（参考 CS336 / minGPT / LLaMA 思路）

核心组件：
- RMSNorm      : Pre-Norm 归一化（比 LayerNorm 更现代，LLaMA 风格）
- RoPE         : 旋转位置编码（相对位置编码，CS336 重点）
- Multi-Head Attention : 多头注意力，支持 KV-cache 加速生成
- SwiGLU FFN   : 门控前馈网络（现代 LLM 标配，比 GELU 效果更好）

训练时走完整序列并行计算；生成时用 KV-cache 逐 token 推理。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 模型配置
# ============================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 50257     # 词表大小（由 tokenizer 决定）
    d_model: int = 256          # 模型宽度
    n_heads: int = 8            # 注意力头数
    n_layers: int = 6           # 层数
    d_ff: int = 512             # FFN 中间维度
    max_seq_len: int = 1024     # 最大序列长度
    dropout: float = 0.1        # dropout 概率
    use_swiglu: bool = True     # 用 SwiGLU 还是普通 GELU FFN


# ============================================================================
# 1. RMSNorm —— 归一化层（Pre-Norm 使用）
#    x_norm = x / RMS(x) * weight
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# 2. RoPE —— 旋转位置编码
#    核心思想：对每个位置 t，把 query/key 按角度 t*theta 旋转。
#    旋转后的注意力分数只依赖相对位置 (t - s)，因此天然具备相对位置编码能力。
# ============================================================================
def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    预计算每个位置的 cos / sin 旋转角。
    返回:
        cos: (max_seq_len, head_dim)
        sin: (max_seq_len, head_dim)

    注意：这里用 torch.cat 把频率拼接成两份（前一半与后一半频率相同），
    而不是 repeat_interleave。这样第 i 维与第 i + d/2 维配对旋转，
    与 rotate_half（前半 vs 后半）的分割方式精确匹配 —— 这是 LLaMA/HF 的标准做法。
    """
    # 频率，维度两两一组: 1/theta^(0), 1/theta^(2/d), ...
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    t = torch.arange(max_seq_len).float()                                        # (max_seq_len,)
    angles = torch.outer(t, freqs)                                               # (max_seq_len, head_dim/2)
    # 拼成两份: [θ0, θ1, ..., θ0, θ1, ...]  → 与 rotate_half 的前后半分割配对
    angles = torch.cat([angles, angles], dim=-1)                                 # (max_seq_len, head_dim)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后一半取负放到前一半前面，实现 90 度旋转: (x1, x2) -> (-x2, x1)"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor):
    """
    对 q / k 应用旋转位置编码。
    q, k: (B, H, T, head_dim)
    cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ============================================================================
# 3. Multi-Head Attention —— 多头注意力（支持 KV-cache）
#    KV-cache 原理：生成时每步只算 1 个新 token，把历史 K/V 缓存起来，
#    避免每步重新计算，显著加速生成。
# ============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim * self.n_heads == cfg.d_model, "d_model 必须能被 n_heads 整除"

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None,
                past_kv: tuple = None):
        """
        x: (B, T, D)  当前输入的 token 表示
        cos/sin: 预计算好的旋转角（整表），内部按当前位置切片
        mask: causal mask (T, T)，训练时用；生成时传 None
        past_kv: (past_k, past_v) 缓存的历史 K/V；None 表示无缓存（训练）
        返回:
            out: (B, T, D)
            new_kv: 更新后的 K/V（供下次生成缓存）
        """
        B, T, _ = x.shape

        # 1. 投影出 q/k/v，拆成多头: (B, H, T, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. 当前 token 的绝对位置起点（有缓存时 = 已有 token 数）
        start = 0 if past_kv is None else past_kv[0].shape[2]
        q, k = apply_rotary_emb(q, k, cos[start:start + T], sin[start:start + T])

        # 3. 拼接历史 K/V（KV-cache 核心）
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)  # 供下一次调用缓存

        # 4. 缩放点积注意力: scores = Q @ K^T / sqrt(d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, T, S)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # 5. 加权求和 + 输出投影
        out = probs @ v                              # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.wo(out)
        return out, new_kv


# ============================================================================
# 4. FeedForward —— 前馈网络
#    SwiGLU: 门控结构，out = W_down( SiLU(x W_gate) ⊙ (x W_up) )
#    相比普通 FFN，SwiGLU 是现代 LLM（LLaMA 等）常用的激活。
# ============================================================================
class FeedForward(nn.Module):
    """普通 GELU FFN（备用）"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class SwiGLU(nn.Module):
    """SwiGLU 门控 FFN（默认）"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# ============================================================================
# 5. DecoderBlock —— 一个 Transformer 解码层（Pre-Norm + 残差）
#    x -> RMSNorm -> MHA -> (+ x) -> RMSNorm -> FFN -> (+ x)
# ============================================================================
class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg) if cfg.use_swiglu else FeedForward(cfg)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None,
                past_kv: tuple = None):
        # Pre-Norm：先归一化再进子层，残差连接保持梯度稳定
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, mask, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ============================================================================
# 6. DecoderOnlyLM —— 完整语言模型
# ============================================================================
class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)                     # 最后的归一化（LLaMA 风格）
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight Tying：输入输出共享 embedding，减少参数、加速收敛
        self.lm_head.weight = self.token_embedding.weight

        # 预计算 RoPE 旋转角并注册为 buffer（随模型保存，但不参与梯度）
        cos, sin = precompute_freqs_cis(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """初始化参数（GPT-2 风格：std=0.02 正态分布）"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, past_kvs: list = None):
        """
        前向传播。
        input_ids: (B, T)  token 序列（训练时整段；生成时每步 1 个 token）
        past_kvs: 各层的 K/V 缓存列表；None 表示训练模式（无缓存）
        返回:
            logits: (B, T, vocab_size)
            new_kvs: 更新后的缓存（供下一步生成）
        """
        B, T = input_ids.shape

        x = self.token_embedding(input_ids)          # (B, T, D)
        x = x * (self.cfg.d_model ** 0.5)            # 缩放 embedding（GPT-2 风格）

        # 训练时构造 causal mask：上三角遮掉未来 token
        mask = None
        if past_kvs is None:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )

        new_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, new_kv = layer(x, self.cos, self.sin, mask, past_kv)
            new_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)
        return logits, new_kvs

    def param_count(self) -> int:
        """模型总参数量（方便写进 README/CV）"""
        return sum(p.numel() for p in self.parameters())
