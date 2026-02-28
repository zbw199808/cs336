# Lecture 09 深入详细学习笔记：Scaling Laws 基础

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 9 - Scaling laws basics.pdf` (53 页幻灯片)

---

## 目录

1. [动机：为什么需要 Scaling Laws](#1-动机为什么需要-scaling-laws)
2. [历史背景：数据 Scaling 的早期研究](#2-历史背景数据-scaling-的早期研究)
3. [数据 vs 性能](#3-数据-vs-性能)
4. [模型工程：用 Scaling Laws 做设计决策](#4-模型工程用-scaling-laws-做设计决策)
5. [数据-模型联合 Scaling](#5-数据-模型联合-scaling)
6. [计算预算权衡与 Chinchilla](#6-计算预算权衡与-chinchilla)

---

## 1. 动机：为什么需要 Scaling Laws

### 1.1 现实场景

> 你的朋友给了你 10,000 块 H100 用一个月，让你训练一个好的开源 LM。你怎么办？

需要做的决策：
1. 搭建基础设施和分布式训练框架 (Assignment 2)
2. 准备高质量的预训练数据集 (Assignment 4)
3. **选择模型大小和训练配置** ← 本讲聚焦于此

### 1.2 Scaling 不简单

- 宽还是深？多少个 head？用什么非线性？
- 可以从现有 LM 抄参数——但最初这些参数是怎么确定的？

### 1.3 Scaling Laws 的承诺

**Scaling Laws = 简单、可预测的模型性能规律**

```
旧方法: 在大模型上调超参数 (昂贵、不可扩展)
新方法: 在小模型上调，用 scaling laws 外推到大模型
```

---

## 2. 历史背景：数据 Scaling 的早期研究

### 2.1 样本复杂度理论

理论家很早就研究 "scaling"——但这些是**上界**，不是实际的 loss 值。

| 理论 | 描述 |
|------|------|
| VC 维 | 有限假设类的学习复杂度 |
| Hall 1989 | k 个假设集合中学习 |
| 密度估计 | 光滑密度的生成模型 |

### 2.2 早期实证研究

| 年份 | 工作 | 发现 |
|------|------|------|
| 1993 | 最早的数据 scaling 论文 | 数据量与性能的关系 |
| 2001 | Banko & Brill | 数据的对数线性 scaling |
| 2012 | Kolachina et al. | 数据与下游性能的**幂律关系** |
| 2017 | Hestness et al. | 首个大规模神经 scaling 研究，多任务 (MT, LM, Speech) |

### 2.3 Hestness et al. 2017 的先见

- 预测了 scaling 的形状
- 讨论了"涌现"(Emergence) 概念
- 提出了按计算量 scaling 和速度-准确率权衡

---

## 3. 数据 vs 性能

### 3.1 幂律关系

**核心发现 [Kaplan+ 2020]**：

```
在 log-log 图上，Loss 和数据集大小呈线性关系
→ 这就是"幂律" (Power Law) 或 "无标度" (Scale-free) 关系
```

$$\text{Error} \propto n^{-\alpha}$$

且这个关系在**多种不同现象**上都成立，甚至在非标准设置 (train ≠ test) 中也成立。

### 3.2 概念基础：为什么是幂律？

**Toy Example: 均值估计**

```
输入: x₁, ..., xₙ ~ N(μ, σ²)
任务: 估计平均值 μ̂ = Σxᵢ/n
误差: E[|μ̂ - μ|²] = σ²/n

→ log(Error) = -log(n) + 2log(σ)
→ 这就是一个 scaling law!
```

更一般地，任何多项式速率 1/n^α 都是一个 scaling law。

### 3.3 指数之谜

经典模型（回归等）的 scaling 是 1/n → 斜率 = -1

但神经网络 scaling laws 的斜率**远小于 -1**：

| 领域 | 观测到的斜率 |
|------|-------------|
| 机器翻译 | 远小于 -1 |
| 语音 | 远小于 -1 |
| 语言建模 | 远小于 -1 |

### 3.4 非参数学习的解释

**核心洞察**：当模型是非参数的（如神经网络），scaling 受**数据内在维度**影响。

```
d 维空间中的函数估计:
  Error ≈ n^(-1/d)

→ 斜率 = -1/d
→ 数据维度越高，scaling 越慢
```

**内在维度理论 [Bahri 2021]**：
1. Scaling laws 来自多项式学习速率 1/n^α
2. 斜率 α 与数据的内在维度密切相关

### 3.5 其他数据 Scaling Laws

**数据组成的影响 [Kaplan+ 2021]**：
- 数据组成影响**偏移量**（截距），而不是**斜率**
- 分布偏移 scaling laws 可以指导多样化数据收集

**数据重复 [Muennighoff et al.]**：
- 有限数据时，重复使用的效用递减
- D' = 有效数据量 < 实际 tokens × 重复次数
- 数据选择应该**自适应于训练规模**

---

## 4. 模型工程：用 Scaling Laws 做设计决策

### 4.1 Scaling Law 驱动的设计流程

```
1. 训练几个小模型
2. 建立 scaling law (如 Adam vs SGD 的 scaling law)
3. 根据 scaling law 预测选择最优超参数
```

### 4.2 架构选择：Transformer vs LSTM

**暴力方法**：花数千万美元训练一个 LSTM GPT-3
**Scaling law 方法**：在小规模比较，外推到大规模 [Kaplan+ 2021]

结论：Transformer 在大规模下明显优于 LSTM。

**跨架构 Scaling [Tay et al.]**：不同架构在小规模的差异可以预测大规模的差异。

### 4.3 优化器选择：Adam vs SGD

[Hestness+ 2017] 的实验（2017 年，pre-Transformer 时代）表明：
- Adam 的 scaling 曲线始终优于 SGD
- 差距在大规模下保持或扩大

### 4.4 深度/宽度权衡

| 发现 | 细节 |
|------|------|
| 1 vs 2 层 | 巨大差异 |
| 更多层 | 在 10⁷ 参数以下收益递减 |
| 非所有参数等价 | Embedding 层参数行为不同 |
| 宽高比 | 是否依赖于规模？待研究 |

### 4.5 Batch Size：临界批量大小

**临界批量大小 (Critical Batch Size)** [McCandlish et al.]：

```
定义: 在给定 loss 目标下的最小有效样本数 / 最小步数

key insight:
- loss 目标越低 → 临界批量大小越大
- 计算量和模型增大时 → 应使用更大的 batch
```

这对**数据并行**的扩展是好消息。

### 4.6 学习率：muP

**问题**：朴素地 scale up 时，最优学习率依赖于模型规模。

**解决方案 [Yang et al. 2022, Yao et al. 2024]**：
- muP (Maximum Update Parametrization)
- 通过规模感知的初始化和学习率缩放，使超参数在不同规模间保持不变

### 4.7 注意：下游表现可能不同

[Tay et al. 2023] 发现：虽然 perplexity scaling 很可预测，但**下游任务 scaling** 可能不那么可预测。

---

## 5. 数据-模型联合 Scaling

### 5.1 核心问题

> 我们应该用更多数据还是更大模型？

大量数据在小模型上浪费了；大模型在少数据上也浪费了。

### 5.2 联合 Scaling Law 公式

**Rosenfeld+ 2020**：
$$\text{Error} = n^{-\alpha} + m^{-\beta} + C$$

**Kaplan+ 2020**：
$$\text{Error} = m^{-\alpha} + n^{-1/\beta}$$

其中 n = 数据量，m = 模型参数量。

### 5.3 预测能力

在小数据、小模型上拟合 scaling 指数，可以**准确预测**大数据、大模型的 error。

```
给定你的成本约束，优化:
  min  n^{-α} + m^{-β} + C
  s.t. cost(n, m) ≤ budget
```

---

## 6. 计算预算权衡与 Chinchilla

### 6.1 核心问题

> 固定计算预算下，训练欠训的大模型 vs 充分训练的小模型？

### 6.2 Kaplan vs Chinchilla 的分歧

Rosenfeld 和 Kaplan 都预测了数据-模型-性能的关系，但 **Chinchilla [Hoffmann et al. 2022]** 认为这些拟合不够准确。

**主要差异**：学习率调度的处理方式不同。

### 6.3 Chinchilla 三种方法

| 方法 | 描述 | 优缺点 |
|------|------|--------|
| **Method 1** | 所有训练曲线的最小包络线是幂律 | 简单但粗糙 |
| **Method 2 (IsoFLOP)** | 固定 FLOP 预算，变化参数量，取最优 | 最直观 |
| **Method 3** | 在大小-数据网格上运行，最小二乘拟合 | 可能有过拟合 |

### 6.4 Method 3 的已知错误

[Besiroglu et al. 2024] 进行了数据取证：
- 恢复原始数据
- 重新拟合得到与 Method 1 和 2 更一致的结果

### 6.5 训练最优 ≠ 部署最优

**关键洞察**：Chinchilla 告诉你固定训练计算下的最优模型，但实际部署中大部分计算在**推理**！

```
应该 "过训练" (overtrain) 以减少推理成本:

模型           tokens/param 比率
GPT-3          2
Chinchilla     20
LLaMA 65B      22
Llama 2 70B    29
Mistral 7B     110
Llama 3 70B    215     ← 远超 Chinchilla 建议！
```

**趋势**：预期使用量越大 → 越值得多花训练成本 → 更高的 tokens/param 比率。

### 6.6 扩展到其他领域

Scaling laws 方法（如 IsoFLOP）同样适用于：
- 扩散模型 [Gulrajani+ 2023]
- 其他生成模型

---

## 总结

### Scaling Laws 为什么令人惊讶且有用？

| 方面 | 价值 |
|------|------|
| 数据 Scaling | 理解数据如何影响模型，有清晰的理论基础 |
| 模型 Scaling | 大幅降低训练成本（小模型实验→外推） |
| 预测能力 | 理解哪些问题可以被"暴力"解决 |

### 设计流程总结

```
1. 在小规模训练多个模型
2. 拟合 scaling law
3. 外推预测大规模最优配置
4. 训练大模型
```

### 关键公式

| 公式 | 含义 |
|------|------|
| Error ∝ n^(-α) | 数据量 n 的幂律 |
| Error = n^(-α) + m^(-β) + C | 数据-模型联合 scaling |
| Critical batch size ∝ 1/target_loss | 批量大小应随目标变化 |
| Tokens/param ∝ expected_usage | 部署导向的训练策略 |

---

下一讲：Lecture 10 — 推理优化
