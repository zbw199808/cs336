# Lecture 03 深入详细学习笔记：LM 架构与训练的一切细节

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 3 - architecture.pdf` (68 页幻灯片)
> 讲师: Tatsu Hashimoto

---

## 目录

1. [概述与方法论](#1-概述与方法论)
2. [Pre-Norm vs Post-Norm](#2-pre-norm-vs-post-norm)
3. [LayerNorm vs RMSNorm](#3-layernorm-vs-rmsnorm)
4. [激活函数与门控机制](#4-激活函数与门控机制)
5. [串行 vs 并行层](#5-串行-vs-并行层)
6. [位置编码详解](#6-位置编码详解)
7. [超参数共识与争议](#7-超参数共识与争议)
8. [正则化：Dropout 与 Weight Decay](#8-正则化dropout-与-weight-decay)
9. [训练稳定性技巧](#9-训练稳定性技巧)
10. [注意力头变体：GQA/MQA](#10-注意力头变体gqamqa)
11. [稀疏与滑动窗口注意力](#11-稀疏与滑动窗口注意力)
12. [总结：现代 LLM 架构共识](#12-总结现代-llm-架构共识)

---

## 1. 概述与方法论

### 1.1 本讲定位

本讲的核心问题：**大型语言模型在架构和训练上有哪些共识？哪些是变化的？我们能从中学到什么？**

方法论：不是理论推导，而是**从数据中学习**——查看过去一年发布的 19+ 个 dense 模型，总结它们的架构选择和超参数设定。

### 1.2 标准 Transformer vs 现代变体

| 组件 | 原始 Transformer (2017) | 现代变体 (2023+) |
|------|------------------------|------------------|
| 归一化位置 | Post-Norm | **Pre-Norm** |
| 归一化类型 | LayerNorm | **RMSNorm** |
| 位置编码 | Sinusoidal | **RoPE** |
| 前馈激活 | ReLU | **SwiGLU** |
| 偏置项 | 有 bias | **无 bias** |

课程 Assignment 1 中实现的正是右侧的现代变体。

### 1.3 今天的主题

> "学习的最好方式是亲手实践；第二好的方式是试图从他人的经验中学习。"

---

## 2. Pre-Norm vs Post-Norm

### 2.1 区别

```
Post-Norm (原始 Transformer):        Pre-Norm (现代标准):
x → Attention → Add → LayerNorm      x → LayerNorm → Attention → Add
    → FFN     → Add → LayerNorm          → LayerNorm → FFN     → Add

区别在于 LayerNorm 是在残差连接之后(post)还是之前(pre)
```

**关键洞察**：Pre-Norm 的设计使得 LayerNorm 不影响残差信号的主路径——归一化在"侧枝"上进行，残差连接保持干净。

### 2.2 为什么 Pre-Norm 更好

1. **梯度衰减 (Gradient Attenuation)**：Xiong 2020 证明 Post-Norm 中梯度在传播过程中会衰减，而 Pre-Norm 不会
2. **梯度尖峰 (Gradient Spikes)**：Post-Norm 更容易出现梯度尖峰
3. **实际好处**：训练更稳定，支持更大的学习率，减少对 warmup 的依赖

### 2.3 共识程度

**几乎所有现代 LLM 都用 Pre-Norm。** 唯一的奇怪例外：OPT-350M 使用了 Post-Norm（原因不明）。

### 2.4 新趋势：Double Norm

最新的模型开始尝试"双重归一化"：

```
标准 Pre-Norm:
x → LayerNorm → Attention → Add(x, ...)

Double Norm (Grok, Gemma 2):
x → LayerNorm → Attention → LayerNorm → Add(x, ...)
                             ↑ 额外的 post-norm (在残差流外面)
```

OLMo 2 只在残差流外部做 post-norm。逻辑是：如果 LayerNorm 在残差流内部是有害的，那在外部加一个 post-norm 应该没问题。

---

## 3. LayerNorm vs RMSNorm

### 3.1 公式对比

**LayerNorm**：
```
y = (x - μ) / √(σ² + ε) * γ + β
```
- 减去均值 μ
- 除以标准差 σ
- 可学习的缩放参数 γ 和偏置 β

**RMSNorm**：
```
y = x / √(mean(x²) + ε) * γ
```
- **不减去均值** (无 μ 计算)
- **无偏置项 β**
- 只做缩放归一化

### 3.2 使用情况

| 使用 LayerNorm | 使用 RMSNorm |
|---------------|-------------|
| GPT-1/2/3, OPT, GPT-J, BLOOM | LLaMA 系列, PaLM, Chinchilla, T5 |

### 3.3 为什么选 RMSNorm？

**表面解释**：更快，参数更少。但需要深入理解为什么"快"——

**FLOPS ≠ 运行时间！** 这是一个极其重要的认知。

矩阵乘法占了绝大多数 FLOPs（和内存），归一化操作的 FLOPs 本身微不足道。但 RMSNorm 的优势在于减少了**数据移动**：

```
[Ivanov et al 2023 的发现]
左图: 43G FLOPs (FLOPs 量)
右图: 153 (FLOP-to-memory ratio)

RMSNorm 虽然节省的 FLOPs 很少，
但减少了内存访问操作，这才是真正的加速来源。
```

**实验验证**：Narang et al 2020 显示 RMSNorm 不仅更快，甚至有时效果更好。

### 3.4 更广泛的趋势：去除偏置项

大多数现代 Transformer **完全不使用偏置项**：

```python
# 原始 Transformer (有偏置):
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

# 现代实现 (无偏置):
FFN(x) = σ(xW₁)W₂
```

原因：
- 偏置项的参数量很少，但需要额外的内存移动
- 去除偏置项有时还能提高优化稳定性

---

## 4. 激活函数与门控机制

### 4.1 激活函数动物园

```
ReLU → GeLU → SwiGLU
 ↓       ↓       ↓
简单    更平滑   门控+平滑

ReLU:    max(0, x)
GeLU:    x · Φ(x)  (Φ 是标准正态的 CDF)
Swish:   x · sigmoid(x)
```

使用情况：

| 激活函数 | 代表模型 |
|----------|----------|
| ReLU | Original Transformer, T5, Gopher, Chinchilla, OPT |
| GeLU | GPT-1/2/3, GPT-J, GPT-NeoX, BLOOM |
| SwiGLU/GeGLU | **LLaMA, PaLM, Mistral, 2023后大多数模型** |

### 4.2 门控线性单元 (GLU) 详解

GLU 的核心思想：在 FFN 的第一部分增加一个**门控机制**。

**标准 FFN**：
```
FFN(x) = ReLU(xW₁) W₂
```

**ReGLU (门控版本)**：
```
FFN_ReGLU(x) = (ReLU(xW₁) ⊗ xV) W₂
                              ↑ 新增的门控项！
```

**SwiGLU**：
```
FFN_SwiGLU(x) = (Swish(xW₁) ⊗ xV) W₂
```

**GeGLU**：
```
FFN_GeGLU(x) = (GeLU(xW₁) ⊗ xV) W₂
```

### 4.3 GLU 的参数代价

门控版本有 **3 个矩阵** (W₁, V, W₂) 而非 2 个。为保持参数量大致相同：

```
标准 FFN:   d_ff = 4 × d_model  (2个矩阵: W₁, W₂)
GLU 变体:   d_ff = 8/3 × d_model ≈ 2.67 × d_model  (3个矩阵: W₁, V, W₂)
```

缩小到 2/3 后总参数量大致持平。

### 4.4 实验证据

Shazeer 2020 和 Narang et al 2020 的实验都显示 GLU 变体带来**一致但不巨大**的性能提升。

**重要结论**：
- *GLU 不是必须的——GPT-3 不用 GLU 也很好
- 但证据指向 SwiGLU/GeGLU 带来一致性的小幅增益
- 异常值模型：Nemotron 340B 用 Squared ReLU，Falcon 2 11B 用 ReLU

---

## 5. 串行 vs 并行层

### 5.1 标准 (串行) Transformer Block

```
x → Attention → Add → FFN → Add → output
```

Attention 和 FFN 是串行执行的。

### 5.2 并行 Transformer Block

```
        ┌→ Attention →┐
x → LN ─┤             ├→ Add → output
        └→ FFN ───────┘
```

Attention 和 FFN **并行执行**。最早由 GPT-J (EleutherAI, 2021) 提出。

**好处**：
- LayerNorm 可以共享（只算一次）
- 矩阵乘法可以融合 (fuse)
- 计算速度更快

**使用模型**：GPT-J, PaLM, GPT-NeoX, Cohere Command A, Falcon 2 11B

**现状**：没有极严格的消融实验，但有计算效率优势。

---

## 6. 位置编码详解

### 6.1 四种方案演进

| 方案 | 原理 | 代表模型 | 时代 |
|------|------|----------|------|
| Sinusoidal | 加法: embed(x,i) = v_x + PE_pos | Original Transformer | 2017 |
| Absolute (学习) | 加法: embed(x,i) = v_x + u_i | GPT-1/2/3, OPT | 2018-2020 |
| Relative (T5式) | 在注意力计算中加入相对位置向量 | T5, Gopher, Chinchilla | 2020-2022 |
| **RoPE** | 旋转: 用旋转矩阵编码位置 | **GPT-J, PaLM, LLaMA, 2024+几乎所有模型** | 2021+ |

### 6.2 RoPE 的核心思想

**目标**：设计一个位置编码函数 f(x, i)，使得注意力分数只依赖于**相对位置 (i-j)**：

```
⟨f(x, i), f(y, j)⟩ = g(x, y, i-j)
```

**为什么之前的方案不满足这个目标？**

- **Sinusoidal**：`⟨v_x + PE_i, v_y + PE_j⟩ = ⟨v_x, v_y⟩ + ⟨PE_i, v_y⟩ + ...`
  - 有交叉项 `⟨PE_i, v_y⟩` 不是相对位置的函数
- **Absolute**：显然不是相对位置编码
- **Relative**：虽然编码了相对位置，但不是通过内积实现的

### 6.3 RoPE 的巧妙设计

**关键洞察**：内积对旋转不变。

如果我们将位置 i 的 token 旋转 θ×i 角度，位置 j 的 token 旋转 θ×j 角度，那么：

```
⟨R(θi)·x, R(θj)·y⟩ = ⟨x, R(θ(j-i))·y⟩
```

内积只依赖于旋转角度之差，即相对位置 (j-i)！

**具体实现**：

将 d_model 维的向量两两配对，在每对上做 2D 旋转：

```
对于维度对 (2k, 2k+1)，旋转角度为 θ_k × position:

[cos(θ_k·i)  -sin(θ_k·i)] [x_2k  ]
[sin(θ_k·i)   cos(θ_k·i)] [x_2k+1]
```

其中不同维度对使用不同的频率 θ_k（类似于 Sinusoidal 编码的多频率设计）。

### 6.4 RoPE vs Sinusoidal 的关键区别

| | Sinusoidal | RoPE |
|---|-----------|------|
| 方式 | 加法 (additive) | 乘法 (multiplicative) |
| 交叉项 | 有 | **无** |
| 相对位置 | 近似 | **精确** |
| 应用位置 | 嵌入层 (一次) | **每层注意力中** (重复) |

### 6.5 实现要点

```python
# 伪代码
cos, sin = get_rope_matrix(positions, dim)   # 预计算旋转矩阵
q_rotated = apply_rotary(q, cos, sin)         # 旋转 query
k_rotated = apply_rotary(k, cos, sin)         # 旋转 key
attn = q_rotated @ k_rotated.T               # 正常计算注意力
```

注意：RoPE 只应用于 query 和 key，**不应用于 value**。

---

## 7. 超参数共识与争议

### 7.1 FFN 维度比

**共识**：`d_ff = 4 × d_model` (非门控) 或 `d_ff ≈ 8/3 × d_model` (GLU 变体)

| 模型 | d_ff / d_model | 备注 |
|------|----------------|------|
| GPT-3, OPT | 4.0 | 标准 |
| PaLM | 4.0 | SwiGLU 但保持了 4x |
| Mistral 7B | 3.5 | GLU |
| LLaMA-2 70B | 3.5 | GLU |
| LLaMA 70B | 2.68 | GLU |
| DeepSeek 67B | 2.68 | GLU |
| T5 v1.1 | 2.5 | GeGLU |

**异常值**：T5 的 11B 模型使用了 d_ff = 65536，d_model = 1024，**64 倍**的比率！但其后续 T5 v1.1 回到了更标准的 2.5 倍。

**经验证据** (Kaplan et al 2020)：在 1-10 倍的范围内，这个超参数对最终性能影响不大——存在一个宽阔的"盆地"。

### 7.2 注意力头维度

**共识**：`head_dim × num_heads = d_model` (1:1 比率)

| 模型 | Num Heads | Head Dim | D_model | Ratio |
|------|-----------|----------|---------|-------|
| GPT-3 | 96 | 128 | 12288 | 1.0 |
| T5 v1.1 | 64 | 64 | 4096 | 1.0 |
| LLaMA-2 | 64 | 128 | 8192 | 1.0 |
| T5 | 128 | 128 | 1024 | **16.0** |
| LaMDA | 128 | 128 | 8192 | **2.0** |
| PaLM | 48 | 258 | 18432 | **1.48** |

大多数模型保持 1:1，但一些 Google 模型偏离了这个比率。

**注意**：Bhojanapalli et al 2020 提出过反对 1:1 的论点，但实际中我们似乎没有看到显著的"低秩瓶颈"问题。

### 7.3 纵横比 (Aspect Ratio)：深还是宽？

**共识**：`d_model / n_layers ≈ 100-200`

| 模型 | d_model / n_layers |
|------|-------------------|
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT-3/OPT/Mistral/Qwen | 128 |
| LLaMA | ~128 |

**考量因素**：
- 极深的模型更难并行化，延迟更高
- 但太宽的模型每层参数太多，也有问题
- 实际选择中**系统工程考量**（如流水线并行的效率）常常决定最终值

**实验证据**：
- Kaplan et al 2020 和 Tay et al 2021 都显示在较宽范围内这个超参数影响不大
- 但极端值（极深或极宽）会明显变差

### 7.4 词表大小

| 类型 | 典型大小 | 模型举例 |
|------|---------|----------|
| 单语言 | 30-50K | GPT-2/3 (50257), LLaMA (32000), T5 (32128) |
| 多语言/生产系统 | 100-250K | mT5 (250000), PaLM (256000), GPT-4 (100276) |

**规律**：单语言模型不需要巨大的词表，但多语言模型需要更大的词表来覆盖更多语言的子词。

---

## 8. 正则化：Dropout 与 Weight Decay

### 8.1 预训练是否需要正则化？

**反对正则化的理由**：
- 数据量巨大（万亿 tokens），远超参数量
- SGD 通常只过一遍语料，很难过拟合

### 8.2 实际做法

| 模型 | Dropout | Weight Decay |
|------|---------|-------------|
| Original Transformer | 0.1 | 0 |
| GPT-2 | 0.1 | 0.1 |
| T5 | 0.1 | 0 |
| GPT-3 | 0.1 | 0.1 |
| T5 v1.1 | **0** | 0 |
| PaLM | **0** | (variable) |
| OPT | 0.1 | 0.1 |
| **LLaMA** | **0** | **0.1** |
| Qwen 14B | 0.1 | 0.1 |

**趋势**：
- 较旧的模型使用 dropout
- **新模型 (2023+) 大多不用 dropout**，只靠 weight decay
- 例外：Qwen 仍然使用 dropout

### 8.3 Weight Decay 的真正作用

Andriushchenko et al 2023 的有趣发现：

- LLM 中 weight decay 的作用**不是防止过拟合**
- 而是与学习率调度 (特别是 cosine schedule) 相互作用，影响**优化动态**
- 本质上更像是一种优化技巧而非正则化

---

## 9. 训练稳定性技巧

### 9.1 问题所在

大规模训练容易出现不稳定——loss 突然飙升 (spike) 然后可能无法恢复。核心问题通常与 **softmax** 相关。

### 9.2 输出 Softmax 稳定性：Z-Loss

问题：softmax 中的指数运算和除零可能导致数值不稳定。

**Z-Loss 技巧** (PaLM 首创)：

```
L_z = log²(Σ exp(z_i))
```

在标准交叉熵 loss 之外加一个额外的 penalty，鼓励 logits 的尺度不要太大。

使用模型：PaLM, Baichuan 2, DCLM, OLMo 2

### 9.3 注意力 Softmax 稳定性：QK Norm

问题：query 和 key 的内积可能非常大，导致 attention softmax 数值不稳定。

**解决方案**：在进入 softmax 之前，对 query 和 key 做 LayerNorm / RMSNorm。

```
attn = softmax(LayerNorm(Q) · LayerNorm(K)^T / √d)
```

使用模型：DCLM, OLMo 2, Gemma 2
来源：最初用于视觉和多模态模型 (Dehghani 2023, IDEFICS, Chameleon)

### 9.4 Logit Soft-Capping

用 tanh 对 logits 进行软截断：

```
logits = cap × tanh(logits / cap)
```

防止 logits 爆炸，但可能对性能有轻微影响。

---

## 10. 注意力头变体：GQA/MQA

### 10.1 标准多头注意力 (MHA) 的推理瓶颈

**训练时**：所有 token 可以并行处理，计算密集型，GPU 利用率高。

**推理解码时**：必须逐步生成 token，需要使用 **KV Cache**。

```
KV Cache 存储了所有已生成 token 的 Key 和 Value

每生成一个新 token:
1. 计算新 token 的 Q, K, V
2. K, V 加入 KV Cache
3. 新 Q 与所有 cached K 做注意力 (O(n) 次运算)
4. 需要从内存读取整个 KV Cache → 内存带宽瓶颈！
```

**算术强度分析**：

训练时：总运算 O(bnd²)，总内存访问 O(bnd + bn²h + d²) → **计算密集型**

解码时：总运算 O(bnd²)，总内存访问 O(bn²d + nd²) → 算术强度 O(n/d + 1/b)⁻¹ → **内存密集型**

KV Cache 的读取成为瓶颈。

### 10.2 Multi-Query Attention (MQA)

**核心思想**：所有头共享**同一组** Key 和 Value，只有 Query 是多头的。

```
标准 MHA:  每个头有独立的 Q, K, V
MQA:       每个头有独立的 Q，但共享 1 组 K, V
```

**好处**：KV Cache 缩小到 1/num_heads → 大幅减少内存读取。

**代价**：Shazeer 2019 发现有微小的 PPL 损失。

### 10.3 Group-Query Attention (GQA)

GQA 是 MHA 和 MQA 之间的折中：

```
MHA:  num_kv_heads = num_heads      (每个头独立 KV)
GQA:  num_kv_heads = num_groups     (几个头共享一组 KV)
MQA:  num_kv_heads = 1              (所有头共享一组 KV)
```

**GQA 的优势**：
- 可以通过 num_kv_heads 这个"旋钮"在表达力和推理效率之间权衡
- Ainslie 2023 的实验：GQA 几乎没有 PPL 损失

**使用模型**：LLaMA 2/3, Mistral, 大多数 2023+ 模型

---

## 11. 稀疏与滑动窗口注意力

### 11.1 问题

全注意力的复杂度是 O(n²)，对长序列来说非常昂贵。

### 11.2 稀疏注意力 (GPT-3)

GPT-3 交替使用稠密注意力层和稀疏注意力层。稀疏模式限制每个 token 只能关注特定位置的 token。

Child et al 2019 的稀疏 Transformer 提出了多种稀疏模式。

### 11.3 滑动窗口注意力 (Sliding Window Attention)

**核心思想**：每个 token 只关注前面固定窗口大小 w 的 token。

```
全注意力:           滑动窗口 (w=3):
1 1 1 1 1           1 0 0 0 0
1 1 1 1 1           1 1 0 0 0
1 1 1 1 1           1 1 1 0 0
1 1 1 1 1           0 1 1 1 0
1 1 1 1 1           0 0 1 1 1
```

**关键洞察**：通过多层堆叠，有效感受野会扩展。如果模型有 L 层，每层窗口为 w，则有效上下文长度为 L × w。

代表模型：Mistral

### 11.4 当前趋势：交错全注意力和局部注意力

最新做法——不是纯粹的滑动窗口，而是**交替使用**：

```
Cohere Command A: 每 4 层中有 1 层是全注意力
LLaMA 4, Gemma: SWA(滑动窗口) + Full(全注意力) 交替
```

- **全注意力层**：捕捉长程依赖
- **滑动窗口层**：高效处理局部信息

一些模型还在滑动窗口层中去掉位置编码 (NoPE)，只在全注意力层用 RoPE。

---

## 12. 总结：现代 LLM 架构共识

### 12.1 高共识项 (几乎所有模型统一)

| 设计决策 | 共识选择 | 确信度 |
|----------|---------|--------|
| 归一化位置 | Pre-Norm | ⭐⭐⭐⭐⭐ |
| 归一化类型 | RMSNorm | ⭐⭐⭐⭐ |
| 偏置项 | 无 bias | ⭐⭐⭐⭐ |
| 位置编码 | RoPE | ⭐⭐⭐⭐ |
| 激活函数 | SwiGLU 或 GeGLU | ⭐⭐⭐ |
| KV 头 | GQA | ⭐⭐⭐ |
| d_ff / d_model | 4x (标准) 或 8/3x (GLU) | ⭐⭐⭐ |

### 12.2 有变化但趋势明确

| 设计决策 | 趋势 |
|----------|------|
| Dropout | 不用 (仅 weight decay) |
| 串行/并行层 | 有些模型并行，但不是主流 |
| 注意力模式 | 全注意力 + 滑动窗口交替 |
| 稳定性技巧 | Z-Loss, QK Norm 越来越普遍 |

### 12.3 主要差异点

模型之间差异最大的三个方面：
1. **位置编码** (虽然 RoPE 已是主流)
2. **激活函数** (SwiGLU vs GeGLU vs 其他)
3. **分词** (词表大小、BPE 实现)

### 12.4 核心认知

> 现代 LLM 的架构在很大程度上是"LLaMA-like"的——Pre-Norm + RMSNorm + RoPE + SwiGLU + GQA + 无 bias。大多数变化都是在这个基础上的微调。

---

## 关键论文速查

| 主题 | 论文/来源 |
|------|----------|
| Pre-Norm vs Post-Norm | Xiong et al 2020, Salazar & Nguyen 2019 |
| RMSNorm | Zhang & Sennrich 2019 |
| SwiGLU | Shazeer 2020 |
| RoPE | Su et al 2021 |
| GQA | Ainslie et al 2023 (Google) |
| MQA | Shazeer 2019 |
| Z-Loss | PaLM (Chowdhery et al 2022) |
| QK Norm | Dehghani et al 2023 |
| FFN 超参数 | Kaplan et al 2020 |
| Dropout 分析 | Narang et al 2020 |
| Weight Decay 分析 | Andriushchenko et al 2023 |
| 滑动窗口注意力 | Mistral (Jiang et al 2023) |
| 稀疏注意力 | Child et al 2019 |

---

下一讲：Lecture 04 — 混合专家模型 (Mixture of Experts)
