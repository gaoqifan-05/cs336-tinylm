# Tiny Decoder-Only Language Model (from scratch)

从零实现的一个小型 Decoder-Only Transformer 语言模型，基于 Stanford **CS336: Language Modeling from Scratch** 课程思路，用 PyTorch 手写核心组件，在 WikiText-2 上完成预训练。单张消费级 GPU（RTX 4060）即可复现。

## ✨ 项目亮点

- **手写核心组件**：RMSNorm、RoPE 旋转位置编码、Multi-Head Attention（带 KV-cache）、SwiGLU FFN、Weight Tying
- **KV-cache 加速生成**：逐 token 自回归推理时缓存历史 K/V，避免重复计算（冒烟测试已证明与全量计算一致）
- **完整训练管线**：混合精度 AMP、Cosine LR schedule + warmup、Weight decay（AdamW）
- **可复现结果**：WikiText-2 验证集 perplexity 从随机 50257 降至 **~568**，附训练曲线

## 📦 环境要求

- Python 3.11
- PyTorch 2.6+（CUDA 12.x）
- 建议：NVIDIA GPU（RTX 3060/4060 或以上，8GB 显存足够）

```bash
conda create -n cs336 python=3.11 -y
conda activate cs336
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tokenizers tqdm matplotlib pandas requests modelscope
```

## 🚀 快速开始

### 1. 准备数据（首次）

数据集会自动下载并缓存到 `data/`（国内网络建议用 hf-mirror，`data.py` 已内置支持手动放置 `train.parquet` / `validation.parquet`）。

### 2. 训练

```bash
# 小规模冒烟测试（几秒）
python src/train.py --max_steps 30 --epochs 1 --batch_size 4 --seq_len 64 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 128

# 正式训练（约 30-60 分钟）
python src/train.py --epochs 1 --batch_size 8 --seq_len 256
```

训练产出：
- `checkpoints/final.pt` — 最终模型权重
- `outputs/training_curves.png` — 训练 loss / LR / 验证 PPL 曲线

### 3. 文本生成

```bash
# greedy 采样
python src/generate.py --prompt "The meaning of life is"

# top-k 采样（更多样）
python src/generate.py --prompt "Once upon a time" --top_k 50 --temperature 0.8
```

### 4. 运行测试

```bash
python src/smoke_test.py   # 模型前向/反向/KV-cache 一致性验证
python src/data.py         # 数据管道自测
```

## 📊 实验结果

| 指标 | 值 |
|---|---|
| 模型参数量 | 16.8M |
| 词表大小 (BPE) | 50,257 |
| 训练集 tokens | 2.2M |
| 验证集 tokens | 238K |
| 训练 loss (最终) | ~6.03 |
| **验证集 Perplexity** | **~568**（随机初始化 50257）|
| 训练设备 | RTX 4060 Laptop (8GB) |

## 📁 项目结构

```
├── src/
│   ├── model.py        # Decoder-Only Transformer（核心）
│   ├── tokenizer.py    # BPE 分词器（GPT-2 风格 ByteLevel）
│   ├── data.py         # WikiText-2 数据管道
│   ├── train.py        # 预训练脚本（AMP + cosine LR + weight decay）
│   ├── generate.py     # KV-cache 文本生成（greedy / top-k）
│   ├── smoke_test.py   # 模型正确性冒烟测试
│   └── kv_cache_demo.py # KV-cache 张量流动可视化
├── data/               # 数据集 + 分词器（gitignore）
├── checkpoints/        # 模型权重（gitignore）
├── outputs/            # 训练曲线（gitignore）
└── requirements.txt
```

## 📚 技术要点速览

- **RoPE**：通过旋转 q/k 编码相对位置，不增加参数，支持更长序列外推
- **Pre-Norm + Residual**：梯度更稳定，训练不需要 warmup 也能收敛
- **Weight Tying**：输入输出共享 embedding，参数减半、收敛更快
- **KV-cache**：生成复杂度从 O(N²) 降到 O(N)

## 🧪 下一步（可选）

- [ ] 超参消融实验（d_model / n_layers 对比 PPL）
- [ ] 在 tiny instruction dataset 上做 SFT 微调
- [ ] 更长序列 / 更多 epoch 提升 PPL

## 🙏 参考

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [minGPT (Andrej Karpathy)](https://github.com/karpathy/minGPT)
- [GPT-2 (Radford et al.)](https://openai.com/research/language-unsupervised)
