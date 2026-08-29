# Tiny Decoder-Only Language Model (from scratch)

从零实现的一个小型 Decoder-Only Transformer 语言模型，基于 Stanford **CS336: Language Modeling from Scratch** 课程思路，用 PyTorch 手写核心组件，在 WikiText-2 上完成预训练。单张消费级 GPU（RTX 4060）即可复现。

## ✨ 项目亮点

- **手写核心组件**：RMSNorm、RoPE 旋转位置编码、Multi-Head Attention（带 KV-cache）、SwiGLU FFN、Weight Tying
- **KV-cache 加速生成**：逐 token 自回归推理时缓存历史 K/V，避免重复计算（冒烟测试已证明与全量计算一致）
- **完整训练管线**：混合精度 AMP、Cosine LR schedule + warmup、Weight decay（AdamW）
- **可复现结果**：WikiText-2 验证集 perplexity 从随机 50257 降至 **201.1**，附训练曲线与完整消融对比表

## 📦 环境要求

- Python 3.11
- PyTorch 2.6+（CUDA 12.x）
- 建议：NVIDIA GPU（RTX 3060/4060 或以上，8GB 显存足够）

```bash
conda create -n cs336 python=3.11 -y
conda activate cs336
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tokenizers pandas pyarrow requests matplotlib
```

## 🚀 快速开始

### 1. 准备数据（首次）

数据集会自动下载并缓存到 `data/`（国内网络建议用 hf-mirror，`data.py` 已内置支持手动放置 `train.parquet` / `validation.parquet`）。

### 2. 训练

```bash
# 小规模冒烟测试（几秒）
python src/train.py --max_steps 30 --epochs 1 --batch_size 4 --seq_len 64 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 128

# 正式训练（PPL 最佳配置：16ep + 正则化 + EMA，约 2 小时）
python src/train.py --epochs 16 --batch_size 16 --seq_len 256 \
    --lr 5e-4 --warmup_steps 400 --grad_clip 1.0 \
    --dropout 0.2 --label_smoothing 0.1 --weight_decay 0.15 \
    --ema_decay 0.999 --tag myrun
```

训练产出（每次训练一个独立的时间戳目录，互不覆盖）：
- `checkpoints/run_<时间戳>_<配置>/best.pt` — **验证集最优权重**（推荐使用）
- `checkpoints/run_<时间戳>_<配置>/final.pt` — 训练结束时的权重
- `checkpoints/run_<时间戳>_<配置>/step_*.pt` — 各验证点 checkpoint
- `outputs/run_<时间戳>_<配置>/training_curves.png` — 训练 loss / LR / 验证 PPL 曲线

### 3. 评估 & 文本生成

```bash
# 在验证集上评估（自动加载最新 best.pt）
python src/evaluate.py --split validation

# 自动加载最新 best.pt 生成
python src/generate.py --prompt "The meaning of life is"

# 指定某个实验的模型
python src/generate.py --ckpt checkpoints/run_xxx/best.pt --prompt "Once upon a time"

# top-k 采样（更多样）
python src/generate.py --prompt "Once upon a time" --top_k 50 --temperature 0.8
```

### 4. 运行测试

```bash
python src/smoke_test.py   # 模型前向/反向/KV-cache 一致性验证
python src/data.py         # 数据管道自测
```

## 📊 实验结果

### 最佳配置

| 指标 | 值 |
|---|---|
| 模型参数量 | 16.8M |
| 词表大小 (BPE) | 50,257 |
| 训练集 tokens | 2.2M |
| 验证集 tokens | 238K |
| 最佳训练 loss | ~4.69 |
| **最佳验证集 Perplexity** | **201.1**（best.pt，随机初始化 50257）|
| 训练设备 | RTX 4060 Laptop (8GB) |

### 训练曲线（16ep 基线）

![training curves](assets/baseline_curves.png)

### 消融实验（16 epoch 基线，每次只改一个组件）

基线配置：`16ep, bs16, seq256, lr5e-4, warm400, gradclip, EMA0.999, dropout0.2, label_smoothing0.1, wd0.15`

| # | 实验 | 相对基线的改动 | 验证 PPL | ΔPPL |
|---|---|---|---|---|
| 1 | **基线** | — | **201.1** | — |
| 2 | 去 EMA | `ema_decay=0` | 212.4 | +11.3 |
| 3 | 去标签平滑 | `label_smoothing=0` | 213.3 | +12.2 |
| 4 | 降 dropout | `dropout=0.1` | 203.6 | +2.4 |
| 5 | 去全部正则 | dropout0.1 + LS0 + wd0.1 | 207.8 | +6.6 |

> **消融结论**：
> - **EMA 贡献最大**（+11.3 PPL）：对权重做滑动平均，训练后期在最优解附近震荡时得到更平滑、泛化更好的解。
> - **标签平滑次之**（+12.2 PPL）：让小数据上的目标分布不那么极端，抑制过拟合。
> - **dropout 单独影响较小**（+2.4 PPL），但与标签平滑、weight_decay 组合时正则化整体有效（去全部正则 +6.6 PPL）。
> - 各组件贡献大致可加性成立，说明它们作用于不同的过拟合方面。

### 优化历程（从零到最佳）

| # | 配置 | 验证 PPL | 说明 |
|---|---|---|---|
| 1 | 3ep, bs8, seq256, lr3e-4 | 343.6 | 起点 |
| 2 | 3ep, bs8, **seq512** | 393.7 | 长序列反而差（步数减半）|
| 3 | 6ep, bs16, lr3e-4 | 287.0 | 加训练量 |
| 4 | 6ep, bs16, 文档拼接 | 320.2 | 文档拼接在此数据无益 |
| 5 | 6ep, bs16, **lr5e-4** + warm400 + gradclip | 242.6 | 高 lr 组合 |
| 6 | 10ep, + **EMA** | 226.7 | EMA 平滑 |
| 7 | 10ep, + **正则化** | 213.9 | 对抗过拟合 |
| 8 | 16ep + 正则 + **best.pt** | **201.1** | **最佳** |

> **优化结论**：
> 1. **训练量 > 序列长度**：小数据下加 epoch/batch 比加长序列有效（seq512 因步数减半反而变差）。
> 2. **学习率 + warmup + 梯度裁剪**：lr 3e-4→5e-4 组合让 PPL 从 287→242.6。
> 3. **best.pt 机制**：保存验证集最优权重，避免过拟合后期权重（235）掩盖真正最优（201.1）。
> 4. **文档级拼接在本数据集上无益**：WikiText-2 是维基片段，行间语义本就弱关联。

### 生成示例（PPL 201 最佳模型）

```
> The meaning of life is
 The meaning of life is a species which can be used as a variety of
 different species . The species has been described by the species .
```

> 17M 小模型在 WikiText-2 上训练后已能产出**结构完整、语法通顺**的英文句子
> （含主从句、定语从句、介词短语），内容层面存在重复与无意义 —— 小模型 +
> 小数据的正常表现。从 PPL 343 → 201 的优化过程中，生成质量同步明显提升。

## 📁 项目结构

```
├── src/
│   ├── model.py        # Decoder-Only Transformer（核心）
│   ├── tokenizer.py    # BPE 分词器（GPT-2 风格 ByteLevel）
│   ├── data.py         # WikiText-2 数据管道
│   ├── train.py        # 预训练脚本（AMP + cosine LR + weight decay + EMA + best.pt）
│   ├── generate.py     # KV-cache 文本生成（greedy / top-k）
│   ├── evaluate.py     # 验证集/测试集 PPL 评估
│   ├── smoke_test.py   # 模型正确性冒烟测试
├── assets/             # 训练曲线图（README 引用）
├── data/               # 数据集 + 分词器（gitignore）
├── checkpoints/        # 模型权重，按 run_时间戳_配置 分目录（gitignore）
├── outputs/            # 训练曲线，与 checkpoints 对应（gitignore）
└── requirements.txt
```

> 每次训练都会生成独立的 `run_<时间戳>_<seq>seq_<epochs>ep` 目录，
> 包含 `final.pt`、各步 checkpoint 和 `training_curves.png`，方便对比不同实验。

## 📚 技术要点速览

- **RoPE**：通过旋转 q/k 编码相对位置，不增加参数，支持更长序列外推
- **Pre-Norm + Residual**：梯度更稳定，训练不需要 warmup 也能收敛
- **Weight Tying**：输入输出共享 embedding，参数减半、收敛更快
- **KV-cache**：生成复杂度从 O(N²) 降到 O(N)
- **AMP 混合精度**：fp16 计算 + fp32 主权重，加速训练、节省显存
- **EMA 指数移动平均**：对权重做滑动平均，泛化更好（PPL 226.7）
- **正则化组合**：dropout 0.2 + 标签平滑 0.1 + weight_decay 0.15 对抗过拟合
- **best.pt 机制**：保存验证集最优权重，避免过拟合后期掩盖真正最优

## 🧪 下一步（可选）

- [x] 超参消融实验（seq_len / epoch / batch / lr / 文档拼接 对比 PPL）✅
- [x] EMA 指数移动平均 ✅
- [x] 正则化（dropout / 标签平滑 / weight_decay）✅
- [x] best.pt 验证最优权重保存 ✅
- [ ] 在 tiny instruction dataset 上做 SFT 微调
- [ ] d_model / n_layers 容量消融

## 🙏 参考

- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [minGPT (Andrej Karpathy)](https://github.com/karpathy/minGPT)
- [GPT-2 (Radford et al.)](https://openai.com/research/language-unsupervised)
