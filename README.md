# Tiny Decoder-Only Language Model (from scratch)

[English] A small decoder-only Transformer language model built from scratch in PyTorch, inspired by Stanford **CS336: Language Modeling from Scratch**. All core components are hand-implemented. Pre-trained on WikiText-2 and fine-tuned with SFT. Reproducible on a single consumer GPU (RTX 4060).

[中文] 从零实现的一个小型 Decoder-Only Transformer 语言模型，基于 Stanford **CS336: Language Modeling from Scratch** 课程思路，用 PyTorch 手写核心组件，在 WikiText-2 上完成预训练与 SFT 微调。单张消费级 GPU（RTX 4060）即可复现。

---

## ✨ Highlights | 项目亮点

- **Hand-written core components | 手写核心组件**：RMSNorm, RoPE rotary positional embedding, Multi-Head Attention (with KV-cache), SwiGLU FFN, Weight Tying
- **KV-cache accelerated generation | KV-cache 加速生成**：caches history K/V during autoregressive inference to avoid recomputation (verified identical to full computation)
- **Full training pipeline | 完整训练管线**：mixed-precision AMP, Cosine LR schedule + warmup, Weight decay (AdamW)
- **SFT fine-tuning | SFT 监督微调**：fine-tuned on 100 instruction pairs, model learns `Question/Answer` format and generalizes to unseen instructions
- **Reproducible results | 可复现结果**：WikiText-2 val perplexity reduced from random 50257 to **183.2** (41.5M model), with training curves and full ablation tables

---

## 📦 Requirements | 环境要求

- Python 3.11
- PyTorch 2.6+（CUDA 12.x）
- NVIDIA GPU recommended (RTX 3060/4060+, 8GB VRAM) | 建议 NVIDIA GPU（RTX 3060/4060 或以上，8GB 显存）


## 🚀 Quick Start | 快速开始

### 1. Prepare data (first time) | 准备数据（首次）

The dataset is automatically downloaded and cached in `data/`. | 数据集会自动下载并缓存到 `data/`。

```bash
# Download WikiText-2 parquet files manually if auto-download fails:
# 如自动下载失败，可手动下载 parquet 到 data/ 目录：
#   train-00000-of-00001.parquet  →  data/train.parquet
#   validation-00000-of-00001.parquet →  data/validation.parquet
python src/data.py  # 数据管道自测 | data pipeline self-test
```

### 2. Training | 训练

```bash
# Quick smoke test (seconds) | 小规模冒烟测试（几秒）
python src/train.py --max_steps 30 --epochs 1 --batch_size 4 --seq_len 64 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 128

# Full training (~1hr) | 正式训练（PPL 最佳配置：16ep + 正则化 + EMA，约 1 小时）
python src/train.py --epochs 16 --batch_size 16 --seq_len 256 \
    --lr 5e-4 --warmup_steps 400 --grad_clip 1.0 \
    --dropout 0.2 --label_smoothing 0.1 --weight_decay 0.15 \
    --ema_decay 0.999 --tag myrun
```

Each run gets a timestamped directory: | 训练产出（每次训练一个独立的时间戳目录，互不覆盖）

- `checkpoints/run_<timestamp>_<config>/best.pt` — **best validation weights (recommended) | 验证集最优权重（推荐使用）**
- `checkpoints/run_<timestamp>_<config>/final.pt` — final weights | 训练结束时的权重
- `checkpoints/run_<timestamp>_<config>/step_*.pt` — per-eval checkpoints | 各验证点 checkpoint
- `outputs/run_<timestamp>_<config>/training_curves.png` — training curves | 训练 loss / LR / 验证 PPL 曲线

### 3. Evaluation & Generation | 评估 & 文本生成

```bash
# Evaluate on validation (auto-loads latest best.pt) | 在验证集上评估（自动加载最新 best.pt）
python src/evaluate.py --split validation

# Generate with latest best.pt | 自动加载最新 best.pt 生成
python src/generate.py --prompt "The meaning of life is"

# Specify a model | 指定某个实验的模型
python src/generate.py --ckpt checkpoints/run_xxx/best.pt --prompt "Once upon a time"

# Top-k sampling (more diverse) | top-k 采样（更多样）
python src/generate.py --prompt "Once upon a time" --top_k 50 --temperature 0.8
```

### 4. SFT Fine-tuning (optional) | SFT 监督微调（可选）

```bash
# Generate instruction data (100 English Q&A) | 生成指令数据（100 条英文问答）
python src/make_sft_data.py

# Fine-tune on pretrained best model (loss masking: only answer part gets loss) | 微调
python src/sft.py --pretrained checkpoints/best_model.pt --epochs 10 --lr 1e-4

# Test fine-tuned model | 测试微调效果（问它问题）
python src/generate.py --ckpt checkpoints/sft/run_xxx/sft_model.pt --prompt "Question: What is the capital of France?\nAnswer:" \
    --top_k 50 --temperature 0.7 --repetition_penalty 1.2
```

> Tip: use `--repetition_penalty 1.2` to suppress repetition loops in small models. | 提示：小模型生成易重复，用 `--repetition_penalty 1.2` 抑制。

### 5. Run tests | 运行测试

```bash
python src/smoke_test.py   # model forward/backward/KV-cache consistency | 模型正确性验证
python src/data.py         # data pipeline self-test | 数据管道自测
```

## 📊 Results | 实验结果

### Best Config | 最佳配置

| Metric | 指标 |
|---|---|
| Params 模型参数量 | 16.8M（d256）/ 41.5M（d512）|
| Vocab (BPE) 词表大小 | 50,257 |
| Train tokens 训练集 | 2.2M |
| Val tokens 验证集 | 238K |
| Best train loss 最佳训练 loss | ~4.69 |
| **Best Val Perplexity** | **183.2**（d512, best.pt，随机初始化 50257）|
| Device 训练设备 | RTX 4060 Laptop (8GB) |

> Note: default config is d_model=256 (16.8M, PPL 201.1); capacity ablation found d_model=512 (41.5M) reaches PPL 183.2. | 注：默认配置为 d_model=256（16.8M，PPL 201.1）；容量消融发现 d_model=512（41.5M）可达 PPL 183.2。

### Training Curves | 训练曲线 (16ep baseline)

![training curves](assets/baseline_curves.png)

### Ablation (one component at a time) | 消融实验

Baseline 基线配置：`16ep, bs16, seq256, lr5e-4, warm400, gradclip, EMA0.999, dropout0.2, label_smoothing0.1, wd0.15`

| # | Experiment 实验 | Change 改动 | PPL | ΔPPL |
|---|---|---|---|---|
| 1 | **Baseline 基线** | — | **201.1** | — |
| 2 | no EMA 去 EMA | `ema_decay=0` | 212.4 | +11.3 |
| 3 | no label smoothing 去标签平滑 | `label_smoothing=0` | 213.3 | +12.2 |
| 4 | lower dropout 降 dropout | `dropout=0.1` | 203.6 | +2.4 |
| 5 | no regularization 去全部正则 | dropout0.1 + LS0 + wd0.1 | 207.8 | +6.6 |

> **Ablation findings 消融结论**:
> - **EMA contributes most 贡献最大**（+11.3 PPL）：smooths weights for better generalization.
> - **label smoothing next 次之**（+12.2 PPL）：suppresses overfitting.
> - **dropout alone small 单独影响较小**（+2.4 PPL），but effective when combined with other regularization.
> - contributions are roughly additive 各组件贡献大致可加性成立。

### Capacity Ablation | 容量消融 (model width, 16 epoch)

| d_model | n_layers | Params 参数量 | best PPL |
|---|---|---|---|
| 128 | 4 | 7.1M | 261.1 |
| 256 | 6 | 16.8M | 201.1 |
| 512 | 6 | 41.5M | **183.2** |

> **Capacity finding 容量结论**：more params → lower PPL. Observed *performance gain with increased model capacity on small corpus*. d512 overfits more, needs stronger regularization.

### Head Count Ablation | Head 数量消融 (same capacity 16.8M)

| n_heads | head_dim | best PPL |
|---|---|---|
| 4 | 64 | 202.4 |
| 8 | 32 | 201.1 |
| 16 | 16 | 201.3 |

> **Head finding Head 结论**：head count barely affects PPL (201~202) at same capacity; 8 heads is sufficient.

### Optimization Journey | 优化历程 (from scratch to best)

| # | Config 配置 | PPL | Note 说明 |
|---|---|---|---|
| 1 | 3ep, bs8, seq256, lr3e-4 | 343.6 | baseline 起点 |
| 2 | 3ep, bs8, **seq512** | 393.7 | longer seq worse 长序列反而差 |
| 3 | 6ep, bs16, lr3e-4 | 287.0 | more training 加训练量 |
| 4 | 6ep, bs16, doc concat 文档拼接 | 320.2 | not helpful 无益 |
| 5 | 6ep, bs16, **lr5e-4** + warm400 + gradclip | 242.6 | higher lr 高 lr |
| 6 | 10ep, + **EMA** | 226.7 | EMA smoothing |
| 7 | 10ep, + **regularization 正则化** | 213.9 | anti-overfit 对抗过拟合 |
| 8 | 16ep + reg + **best.pt** | **201.1** | **best 最佳** |

> **Optimization findings 优化结论**:
> 1. **training amount > seq length 训练量 > 序列长度**：more epochs/batch beats longer sequences on small data.
> 2. **lr + warmup + grad clip**：lr 3e-4→5e-4 combo dropped PPL 287→242.6.
> 3. **best.pt mechanism**：saves best val weights, avoiding late-overfitting weights hiding the true optimum.

### Supervised Fine-Tuning | SFT 监督微调

Fine-tune on 100 hand-made English Q&A pairs (geography/history/science/biology) to learn the `Question/Answer` format. | 在预训练基础上，用 100 条自制的英文问答对（地理/历史/科学/生物等百科主题）微调，让模型学会 `Question/Answer` 问答格式。

**Core 核心：loss masking** — only the `Answer` part gets loss (the question is input, not to be predicted) | 只对 `Answer` 部分计算 loss（问题是输入，不该被预测）。

#### Before vs After | 微调前后对比 (16.8M model)

| Input 输入 | Before 微调前 | After 微调后 |
|---|---|---|
| `Question: What is the capital of France?` | meaningless continuation 无意义续写 `"the most great @-@ great..."` | `The capital of France is Paris.` ✅ |
| `Question: What is the capital of Germany?`（unseen 未见）| — | `The capital of Germany is Paris.`（format correct, content hallucination 格式正确，内容幻觉）|

> SFT makes the model answer in Q&A format, and the format **generalizes to unseen instructions**. | SFT 让模型从"自由续写"变为"按问答格式回答"，且格式能泛化到未见过的指令。
> Content hallucination due to limited data — normal for small models. | 内容幻觉是因为 100 条数据太少，属小模型正常局限。

#### SFT loss & overfitting | SFT loss 与过拟合 (16.8M vs 41.5M)

| Model 模型 | SFT loss (10ep) | Unseen Q 对未见问题 |
|---|---|---|
| 16.8M | 1.39 | format correct 格式正确 |
| 41.5M | 0.27 | format correct; more accurate on trained Q |

> **Overfitting finding 过拟合观察**：extending 41.5M SFT from 10ep to 16ep dropped loss 0.27→0.035 (memorizing training data), but no improvement on unseen questions and more repetition. **Lower SFT loss ≠ better generalization**; 10ep is enough, 16ep starts overfitting.

### Generation Samples | 生成示例 (PPL 201 best model)

```
> The meaning of life is
 The meaning of life is a species which can be used as a variety of
 different species . The species has been described by the species .

> In the beginning
 In the beginning of this period , Wheeler played a role with the club
 in a British professional football game , a professional club record
 of four matches . In 2008 , he signed with the First Division Player
 of the Year to a 2 – 1 lead in a five @-@ year contract . In early
 2013 , he was named the league 's highest appearance in the Division
 I 'll be known for the Conference , which would be
```

> The 17M model produces grammatically correct English (subordinate clauses, years), with repetition and nonsense content — normal for small models. | 17M 小模型能产出结构完整、语法通顺的英文句子，内容层面存在重复与无意义 —— 小模型的正常表现。

## 📁 Project Structure | 项目结构

```
├── src/
│   ├── model.py         # Decoder-Only Transformer (core 核心)
│   ├── tokenizer.py     # BPE tokenizer (GPT-2 style ByteLevel) 分词器
│   ├── data.py          # WikiText-2 data pipeline 数据管道
│   ├── train.py         # pretraining 预训练 (AMP + cosine LR + EMA + best.pt)
│   ├── generate.py      # KV-cache generation (greedy / top-k / repetition penalty) 生成
│   ├── evaluate.py      # PPL evaluation 评估
│   ├── sft.py           # SFT fine-tuning (loss masking) 监督微调
│   ├── make_sft_data.py # SFT instruction data generation 生成指令数据
│   ├── smoke_test.py    # model correctness smoke tests 冒烟测试
├── assets/              # training curves (referenced by README) 训练曲线
├── data/                # dataset + tokenizer + SFT data (gitignore) 数据集
├── checkpoints/         # model weights, per-run directories (gitignore) 模型权重
├── outputs/             # training curves, matching checkpoints (gitignore) 训练曲线
└── requirements.txt
```

> Each training run creates an independent `run_<timestamp>_<seq>seq_<epochs>ep` directory for easy experiment comparison. | 每次训练都会生成独立目录，方便对比不同实验。

## 📚 Technical Highlights | 技术要点速览

- **RoPE**：rotary positional encoding via q/k rotation, no extra params, supports longer sequences | 旋转位置编码
- **Pre-Norm + Residual**：more stable gradients | 梯度更稳定
- **Weight Tying**：shared input/output embedding, halves params, faster convergence | 共享 embedding
- **KV-cache**：generation complexity from O(N²) to O(N) | 生成加速
- **AMP mixed precision**：fp16 compute + fp32 master weights | 混合精度
- **EMA**：weight moving average, better generalization | 指数移动平均
- **Regularization combo**：dropout 0.2 + label smoothing 0.1 + weight_decay 0.15 | 正则化组合
- **best.pt mechanism**：saves best val weights, avoids late-overfitting | 保存验证最优权重

## 🙏 References | 参考

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [minGPT (Andrej Karpathy)](https://github.com/karpathy/minGPT)
- [GPT-2 (Radford et al.)](https://openai.com/research/language-unsupervised)
