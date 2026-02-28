# Lecture 11 深入详细学习笔记：Scaling 实践案例与细节

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 11 - Scaling details.pdf` (55 页幻灯片)

---

## 目录

1. [动机与概述](#1-动机与概述)
2. [CerebrasGPT：Chinchilla + muP](#2-cerebrasgpt-chinchilla--mup)
3. [MiniCPM：精细的 Scaling 分析](#3-minicpm精细的-scaling-分析)
4. [DeepSeek：直接估计最优超参数](#4-deepseek直接估计最优超参数)
5. [其他近期 Scaling 实践](#5-其他近期-scaling-实践)
6. [muP 深入理解与验证](#6-mup-深入理解与验证)

---

## 1. 动机与概述

### 1.1 核心问题

实践中 scaling 的最佳实践是什么？

1. **Chinchilla 的方法真的有效吗？**
2. **能否节省拟合 scaling law 的计算？**
3. **应该选择什么参数化方式使 scaling 更平滑？**

### 1.2 有详细公开 Scaling 信息的模型

| 模型 | 年份 | 特点 |
|------|------|------|
| CerebrasGPT | 2023 | muP + Chinchilla 公式 |
| MiniCPM | 2024 | muP + 精细 Chinchilla 分析 |
| DeepSeek | 2024 | 直接超参数搜索 + IsoFLOP |
| LLaMA 3 | 2024 | IsoFLOP 分析 |
| Hunyuan-1 | 2024 | MoE 的 IsoFLOP |
| MiniMax-01 | 2025 | 架构 Scaling + Chinchilla Method 1 |

---

## 2. CerebrasGPT：Chinchilla + muP

### 2.1 概述

0.1B 到 13B 的模型，使用 Chinchilla recipe 训练。

**核心发现**：使用 muP 参数化使 scaling 更稳定。

### 2.2 muP 参数化实现

与标准参数化 (SP) 的关键差异：

| 组件 | 标准参数化 (SP) | muP |
|------|----------------|-----|
| 初始化 | 1/√(fan_in) | 依赖 fan_in 和 fan_out |
| 学习率 | 固定 | 随宽度缩放 |
| Embedding scale | 1 | ~10-12 |

**CerebrasGPT 配置**：
```
scale_emb = 10, lr = 6e-3, init_base = 0.08
```

### 2.3 超参数 Scaling 策略

使用 muP 后，超参数在不同规模间更稳定 → 从小模型可以直接迁移。

---

## 3. MiniCPM：精细的 Scaling 分析

### 3.1 概述

清华团队 2024 年发布的小型高性能 LM。

**成就**：1-2.5B 参数模型超越多数同规模模型，匹配许多 7B 模型。

### 3.2 技术 1：muP 稳定 Scaling

```
MiniCPM:     scale_emb = 12, scale_depth = 1.4, init_std = 0.1, lr = 0.01
CerebrasGPT: scale_emb = 10, lr = 6e-3, init_base = 0.08
```

### 3.3 Scaling 策略

```
1. 用 muP 初始化
2. 固定宽高比 (aspect ratio)
3. 逐步增大模型规模
4. 直接拟合最优 batch size、LR、token-to-size 比例
```

**注意**：最大的 scaling 实验模型与实际训练模型之间有 ~5× 的 gap。

### 3.4 最优 Batch Size

三个模型规模 (9M, 30M, 170M) 的实验：
- 纵轴: 数据量, 横轴: batch size, 颜色: loss
- 红线标识每个数据量/模型规模下的最优 batch size
- **结论**：随 loss 降低，最优 batch size 多项式增长

### 3.5 最优学习率

按照 muP 理论，最优 LR 应该基本稳定。实验大致验证了这一点。

### 3.6 WSD 学习率调度

**关键创新**：用 WSD (Warmup-Stable-Decay) 替代 cosine schedule。

```
标准 cosine: 必须从头训练到结束才能得到每个配置的最终 loss
         → 拟合 Chinchilla 的成本从 n 变成 n²

WSD: 分三阶段
  1. Warmup: 快速增加 LR
  2. Stable: 保持 LR 不变
  3. Decay: 快速降低 LR (~10% 的训练步数)

好处: 可以在 stable 阶段结束后重新启动 decay
     → 用一次训练得到多个数据点！
```

### 3.7 Chinchilla 分析

使用 WSD 后，MiniCPM 可以高效地进行 Chinchilla 分析：

| 方法 | 使用情况 |
|------|---------|
| Method 1 (Lower envelope) | 相当清晰的趋势 |
| Method 3 (Joint fit) | 主要方法；发现很高的数据-模型比 |

**关键发现**：数据-模型比率约 192:1，远高于 Chinchilla 的 20:1。

---

## 4. DeepSeek：直接估计最优超参数

### 4.1 概述

7B 和 67B 参数模型，性能大致匹配 LLaMA 2。

### 4.2 Scaling 策略

**与 CerebrasGPT/MiniCPM 的区别**：不用 muP，直接搜索最优超参数。

```
1. 小规模训练 + 收集"近最优"模型 (在 min 的 0.25% 以内)
2. 拟合 LR 随规模的变化 (虽然拟合看起来有些可疑)
3. 用 IsoFLOP 分析选择模型大小
```

### 4.3 WSD 风格学习率

DeepSeek 也使用类似 WSD 的学习率：快速 warmup + 两次 10% 的 decay。

效果：基本匹配 cosine schedule。

### 4.4 Chinchilla Method 2 (IsoFLOP)

直接的 IsoFLOP 分析来选择数据-模型规模的权衡。

**Scaling 预测**：拟合的 scaling 模型能准确预测最终模型的 loss。

---

## 5. 其他近期 Scaling 实践

### 5.1 LLaMA 3 (2024)

- IsoFLOP 风格 scaling：39:1 的数据-参数比
- 计算到下游性能的 scaling

### 5.2 Hunyuan-1 (2024)

- MoE 模型的 IsoFLOP scaling
- 最优比率：96:1（数据到活跃参数）

### 5.3 MiniMax-01 (2025)

- 架构 scaling laws + Chinchilla Method 1

### 5.4 方法对比

| 方法 | CerebrasGPT | MiniCPM | DeepSeek | LLaMA 3+ |
|------|------------|---------|----------|----------|
| 超参数稳定性 | muP | muP | 直接搜索 | 未公开 |
| LR schedule | Cosine | WSD | WSD | 未公开 |
| Chinchilla | 公式 | Method 1+3 | Method 2 | IsoFLOP |

---

## 6. muP 深入理解与验证

### 6.1 muP 的两个条件

给定网络宽度 n_l：

**A1**：初始化时激活值应保持 Θ(1)
**A2**：一次梯度步后，激活值的变化应保持 Θ(1)

注意：如果单个激活值是 Θ(1)，则范数应为 Θ(√n_l)。

### 6.2 推导 muP (条件 A1)

对于深度线性网络 h_l = W_l h_{l-1}，初始化 W_l ~ N(0, σI)：

```
||W_l||* → σ(√n_{l-1} + √n_l)

选择 σ = Θ(1/√n_{l-1} · min(1, √(n_l/n_{l-1})))

归纳:
  ||h_{l-1}||₂ = Θ(√n_{l-1})   (假设)
  ||W_l||*     = √(n_l/n_{l-1}) (结论)
  ||h_l||₂     = Θ(√n_l)        (验证)
```

### 6.3 推导 muP (条件 A2)

SGD 更新：ΔW_l = -η_l · ∇_{h_l}ℓ · h_{l-1}ᵀ

```
Δh_l = W_l·Δh_{l-1} + ΔW_l·(h_{l-1} + Δh_{l-1})

三项都要 Θ(√n_l):
1. W_l·Δh_{l-1} = Θ(√n_l)        (由 A1 和归纳假设)
2. ΔW_l·h_{l-1}                   (需要 ||ΔW_l||* = Θ(√(n_l/n_{l-1})))
3. ΔW_l·Δh_{l-1}                  (高阶项)

结论: η_l = Θ(n_l/n_{l-1})   [SGD]
      η_l = Θ(1/n_{l-1})     [Adam]
```

### 6.4 muP vs 标准参数化

| | 标准参数化 (SP) | muP |
|---|----------------|-----|
| 初始化 σ | 1/√n_{l-1} | Θ(1/√n_{l-1} · min(1, √(n_l/n_{l-1}))) |
| LR (SGD) | Θ(1) | Θ(n_l/n_{l-1}) |
| LR (Adam) | Θ(1) | Θ(1/n_{l-1}) |

### 6.5 muP 的鲁棒性验证

**muP 鲁棒的方面**：

| 因素 | 结果 |
|------|------|
| SwiGLU / Squared ReLU | 最优 LR 保持一致 |
| 不同 Batch size | 基本稳定 |
| Zero Query 初始化 | 保持一致 |
| SP Unembedding | 保持一致 |

**muP 不鲁棒的方面**：

| 因素 | 问题 |
|------|------|
| **RMSNorm learnable gains** | 打破 muP（但移除 gains 对性能影响小） |
| **奇异优化器 (如 Lion)** | 基于梯度符号的优化器不迁移 |
| **强 weight decay (0.1)** | 唯一显著的 muP 失败 |

### 6.6 muP 总结

总体而言 muP 是有用的：
- SP 明显更不稳定
- muP 使超参数更容易调优
- 但需要注意一些已知的不兼容组件

---

## 总结

### 实践中 Scaling 的挑战

| 挑战 | 解决方案 |
|------|---------|
| 设置模型架构超参数 | 假设稳定性 / 使用 muP |
| 设置优化器超参数 | 小规模搜索，固定或预测 scaling |
| 拟合 Chinchilla 的计算成本 | 使用 WSD 学习率调度 |

### 关键教训

1. **muP 使 LR 跨规模保持稳定** → 减少大规模实验需求
2. **WSD 学习率** → 大幅降低 Chinchilla scaling 分析成本
3. **IsoFLOP 方法** 是最常用的 scaling 分析手段
4. **实际的 data:param 比率** 远超 Chinchilla 建议的 20:1

---

下一讲：Lecture 12 — 评估 (Evaluation)
