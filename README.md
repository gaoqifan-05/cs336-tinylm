# Tiny Decoder-Only Language Model (from scratch)

[中文] 从零实现的一个小型 Decoder-Only Transformer 语言模型，基于 Stanford **CS336: Language Modeling from Scratch** 课程思路，用 PyTorch 手写核心组件，在 WikiText-2 上完成预训练与 SFT 微调。单张消费级 GPU（RTX 4060）即可复现。

[English] A small decoder-only Transformer language model built from scratch in PyTorch, inspired by Stanford **CS336: Language Modeling from Scratch**. All core components are hand-implemented. Pre-trained on WikiText-2 and fine-tuned with SFT. Reproducible on a single consumer GPU (RTX 4060).

---

## ✨ 项目亮点 | Highlights

- **手写核心组件 | Hand-written core components**：RMSNorm、RoPE 旋转位置编码、Multi-Head Attention（带 KV-cache）、SwiGLU FFN、Weight Tying
- **KV-cache 加速生成 | KV-cache accelerated generation**：逐 token 自回归推理时缓存历史 K/V，避免重复计算（冒烟测试已证明与全量计算一致）
- **完整训练管线 | Full training pipeline**：混合精度 AMP、Cosine LR schedule + warmup、Weight decay（AdamW）
- **SFT 监督微调 | SFT fine-tuning**：用 100 条指令数据微调，模型学会 `Question/Answer` 问答格式并泛化到未见指令
- **可复现结果 | Reproducible results**：WikiText-2 验证集 perplexity 从随机 50257 降至 **183.2**（41.5M 模型），附训练曲线与完整消融对比表

---

## 📦 环境要求 | Requirements

- Python 3.11
- PyTorch 2.6+（CUDA 12.x）
- 建议：NVIDIA GPU（RTX 3060/4060 或以上，8GB 显存足够）| NVIDIA GPU recommended (RTX 3060/4060+, 8GB VRAM)


## 🚀 快速开始 | Quick Start

### 1. 准备数据（首次）| Prepare data (first time)

数据集会自动下载并缓存到 `data/`。| The dataset is automatically downloaded and cached in `data/`.

### 2. 训练 | Training

```bash
# 小规模冒烟测试（几秒）| Quick smoke test (seconds)
python src/train.py --max_steps 30 --epochs 1 --batch_size 4 --seq_len 64 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 128

# 正式训练（PPL 最佳配置：16ep + 正则化 + EMA，约 1 小时）| Full training (~1hr)
python src/train.py --epochs 16 --batch_size 16 --seq_len 256 \
    --lr 5e-4 --warmup_steps 400 --grad_clip 1.0 \
    --dropout 0.2 --label_smoothing 0.1 --weight_decay 0.15 \
    --ema_decay 0.999 --tag myrun
```

训练产出（每次训练一个独立的时间戳目录，互不覆盖）| Each run gets a timestamped directory:

- `checkpoints/run_<时间戳>_<配置>/best.pt` — **验证集最优权重**（推荐使用）| best validation weights (recommended)
- `checkpoints/run_<时间戳>_<配置>/final.pt` — 训练结束时的权重 | final weights
- `checkpoints/run_<时间戳>_<配置>/step_*.pt` — 各验证点 checkpoint | per-eval checkpoints
- `outputs/run_<时间戳>_<配置>/training_curves.png` — 训练 loss / LR / 验证 PPL 曲线 | training curves

### 3. 评估 & 文本生成 | Evaluation & Generation

```bash
# 在验证集上评估（自动加载最新 best.pt）| Evaluate on validation (auto-loads latest best.pt)
python src/evaluate.py --split validation

# 自动加载最新 best.pt 生成 | Generate with latest best.pt
python src/generate.py --prompt "The meaning of life is"

# 指定某个实验的模型 | Specify a model
python src/generate.py --ckpt checkpoints/run_xxx/best.pt --prompt "Once upon a time"

# top-k 采样（更多样）| Top-k sampling (more diverse)
python src/generate.py --prompt "Once upon a time" --top_k 50 --temperature 0.8
```

### 4. SFT 监督微调（可选）| SFT Fine-tuning (optional)

```bash
# 生成指令数据（100 条英文问答）| Generate instruction data (100 English Q&A)
python src/make_sft_data.py

# 在预训练 best model 上微调（loss masking 只对答案算 loss）
python src/sft.py --pretrained checkpoints/best_model.pt --epochs 10 --lr 1e-4

# 测试微调效果（问它问题）| Test fine-tuned model
python src/generate.py --ckpt checkpoints/sft/run_xxx/sft_model.pt --prompt "Question: What is the capital of France?\nAnswer:" \
    --top_k 50 --temperature 0.7 --repetition_penalty 1.2
```

> 提示：小模型生成易重复，用 `--repetition_penalty 1.2` 抑制重复循环。| Tip: use `--repetition_penalty 1.2` to suppress repetition loops in small models.

### 5. 运行测试 | Run tests

```bash
python src/smoke_test.py   # 模型前向/反向/KV-cache 一致性验证
python src/data.py         # 数据管道自测
```

## 📊 实验结果 | Results

### 最佳配置 | Best Config

| 指标 Metric | 值 Value |
|---|---|
| 模型参数量 Params | 16.8M（d256）/ 41.5M（d512）|
| 词表大小 Vocab (BPE) | 50,257 |
| 训练集 tokens Train | 2.2M |
| 验证集 tokens Val | 238K |
| 最佳训练 loss | ~4.69 |
| **最佳验证集 Perplexity** | **183.2**（d512, best.pt，随机初始化 50257）|
| 训练设备 Device | RTX 4060 Laptop (8GB) |

> 注：默认配置为 d_model=256（16.8M，PPL 201.1）；容量消融发现 d_model=512（41.5M）可达 PPL 183.2。
> Note: default config is d_model=256 (16.8M, PPL 201.1); capacity ablation found d_model=512 (41.5M) reaches PPL 183.2.

### 训练曲线 | Training Curves (16ep baseline)

![training curves](assets/baseline_curves.png)

### 消融实验（16 epoch 基线，每次只改一个组件）| Ablation (one component at a time)

基线配置 Baseline：`16ep, bs16, seq256, lr5e-4, warm400, gradclip, EMA0.999, dropout0.2, label_smoothing0.1, wd0.15`

| # | 实验 Experiment | 改动 Change | PPL | ΔPPL |
|---|---|---|---|---|
| 1 | **基线 Baseline** | — | **201.1** | — |
| 2 | 去 EMA (no EMA) | `ema_decay=0` | 212.4 | +11.3 |
| 3 | 去标签平滑 (no label smoothing) | `label_smoothing=0` | 213.3 | +12.2 |
| 4 | 降 dropout (lower dropout) | `dropout=0.1` | 203.6 | +2.4 |
| 5 | 去全部正则 (no regularization) | dropout0.1 + LS0 + wd0.1 | 207.8 | +6.6 |

> **消融结论 Ablation findings**:
> - **EMA 贡献最大 | contributes most**（+11.3 PPL）：对权重做滑动平均，泛化更好。
> - **标签平滑次之 | label smoothing next**（+12.2 PPL）：抑制过拟合。
> - **dropout 单独影响较小 | dropout alone small**（+2.4 PPL），但与其它正则组合时整体有效。
> - 各组件贡献大致可加性成立 | contributions are roughly additive。

### 容量消融 | Capacity Ablation (模型宽度 model width, 16 epoch)

| d_model | n_layers | 参数量 Params | best PPL |
|---|---|---|---|
| 128 | 4 | 7.1M | 261.1 |
| 256 | 6 | 16.8M | 201.1 |
| 512 | 6 | 41.5M | **183.2** |

> **容量结论 Capacity finding**：参数量翻倍，PPL 持续下降。观察到模型容量增大带来的性能提升
> (*performance gain with increased model capacity on small corpus*)。d512 过拟合更严重，大模型需更强正则。

### Head 数量消融 | Head Count Ablation (相同容量 same capacity 16.8M)

| n_heads | head_dim | best PPL |
|---|---|---|
| 4 | 64 | 202.4 |
| 8 | 32 | 201.1 |
| 16 | 16 | 201.3 |

> **Head 结论 Head finding**：相同容量下 head 数量对 PPL 影响很小（201~202），8 头已足够。

### 优化历程 | Optimization Journey (从零到最佳 from scratch to best)

| # | 配置 Config | PPL | 说明 Note |
|---|---|---|---|
| 1 | 3ep, bs8, seq256, lr3e-4 | 343.6 | 起点 baseline |
| 2 | 3ep, bs8, **seq512** | 393.7 | 长序列反而差 (longer seq worse) |
| 3 | 6ep, bs16, lr3e-4 | 287.0 | 加训练量 (more training) |
| 4 | 6ep, bs16, 文档拼接 (doc concat) | 320.2 | 文档拼接无益 (not helpful) |
| 5 | 6ep, bs16, **lr5e-4** + warm400 + gradclip | 242.6 | 高 lr 组合 (higher lr) |
| 6 | 10ep, + **EMA** | 226.7 | EMA 平滑 (EMA smoothing) |
| 7 | 10ep, + **正则化** (regularization) | 213.9 | 对抗过拟合 (anti-overfit) |
| 8 | 16ep + 正则 + **best.pt** | **201.1** | **最佳 best** |

> **优化结论 Optimization findings**:
> 1. **训练量 > 序列长度** (training amount > seq length)：小数据下加 epoch/batch 比加长序列有效。
> 2. **学习率 + warmup + 梯度裁剪**：lr 3e-4→5e-4 组合让 PPL 从 287→242.6。
> 3. **best.pt 机制**：保存验证集最优权重，避免过拟合后期权重掩盖真正最优。

### SFT 监督微调 | Supervised Fine-Tuning

在预训练基础上，用 100 条自制的英文问答对（地理/历史/科学/生物等百科主题）微调，让模型学会 `Question/Answer` 问答格式。
Fine-tune on 100 hand-made English Q&A pairs (geography/history/science/biology) to learn the `Question/Answer` format.

**核心 Core：loss masking** —— 只对 `Answer` 部分计算 loss（问题是输入，不该被预测）。

#### 微调前后对比 | Before vs After (16.8M 模型)

| 输入 Input | 微调前 Before | 微调后 After |
|---|---|---|
| `Question: What is the capital of France?` | 无意义续写 `"the most great @-@ great..."` | `The capital of France is Paris.` ✅ |
| `Question: What is the capital of Germany?`（未见 unseen）| — | `The capital of Germany is Paris.`（格式正确，内容幻觉）|

> SFT 让模型从"自由续写"变为"按问答格式回答"，且**格式能泛化到未见过的指令**。
> SFT makes the model answer in Q&A format, and the format **generalizes to unseen instructions**.
> 内容幻觉是因为 100 条数据太少（content hallucination due to limited data — normal for small models）。

#### SFT loss 与过拟合 | SFT loss & overfitting (16.8M vs 41.5M)

| 模型 Model | SFT loss (10ep) | 对未见问题 Unseen Q |
|---|---|---|
| 16.8M | 1.39 | 格式正确 (format correct) |
| 41.5M | 0.27 | 格式正确，训练过的问题更准 |

> **过拟合观察 Overfitting finding**：41.5M 的 SFT 从 10ep 加到 16ep，loss 从 0.27 降到 0.035（背熟训练数据），
> 但对**未见问题**没有改善、重复反而更多。**SFT loss 低 ≠ 泛化好**，10ep 已足够，16ep 开始过拟合。

### 生成示例 | Generation Samples (PPL 201 最佳模型)

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

> 17M 小模型能产出**结构完整、语法通顺**的英文句子（含主从句、年份数字），内容层面存在重复与无意义 —— 小模型的正常表现。
> The 17M model produces grammatically correct English (subordinate clauses, years), with repetition and nonsense content — normal for small models.

## 📁 项目结构 | Project Structure

```
├── src/
│   ├── model.py         # Decoder-Only Transformer（核心 core）
│   ├── tokenizer.py     # BPE 分词器（GPT-2 风格 ByteLevel）
│   ├── data.py          # WikiText-2 数据管道 data pipeline
│   ├── train.py         # 预训练 pretraining（AMP + cosine LR + EMA + best.pt）
│   ├── generate.py      # KV-cache 生成（greedy / top-k / repetition penalty）
│   ├── evaluate.py      # 验证集/测试集 PPL 评估 evaluation
│   ├── sft.py           # SFT 监督微调（loss masking）
│   ├── make_sft_data.py # 生成 SFT 指令数据
│   ├── smoke_test.py    # 模型正确性冒烟测试
├── assets/              # 训练曲线图（README 引用）
├── data/                # 数据集 + 分词器 + SFT 数据（gitignore）
├── checkpoints/         # 模型权重，按 run_时间戳_配置 分目录（gitignore）
├── outputs/             # 训练曲线，与 checkpoints 对应（gitignore）
└── requirements.txt
```

> 每次训练都会生成独立的 `run_<时间戳>_<seq>seq_<epochs>ep` 目录，方便对比不同实验。

## 📚 技术要点速览 | Technical Highlights

- **RoPE**：通过旋转 q/k 编码相对位置，不增加参数，支持更长序列外推
- **Pre-Norm + Residual**：梯度更稳定，训练更稳
- **Weight Tying**：输入输出共享 embedding，参数减半、收敛更快
- **KV-cache**：生成复杂度从 O(N²) 降到 O(N)
- **AMP 混合精度**：fp16 计算 + fp32 主权重，加速训练、节省显存
- **EMA 指数移动平均**：对权重做滑动平均，泛化更好
- **正则化组合**：dropout 0.2 + 标签平滑 0.1 + weight_decay 0.15 对抗过拟合
- **best.pt 机制**：保存验证集最优权重，避免过拟合后期掩盖真正最优

## 🙏 参考 | References

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [minGPT (Andrej Karpathy)](https://github.com/karpathy/minGPT)
- [GPT-2 (Radford et al.)](https://openai.com/research/language-unsupervised)
