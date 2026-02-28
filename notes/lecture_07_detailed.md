# Lecture 07 深入详细学习笔记：并行策略基础

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 7 - Parallelism basics.pdf` (60 页幻灯片)

---

## 目录

1. [为什么需要多 GPU 并行](#1-为什么需要多-gpu-并行)
2. [集合通信原语](#2-集合通信原语)
3. [数据并行 (Data Parallelism)](#3-数据并行-data-parallelism)
4. [ZeRO 优化：Stage 1/2/3](#4-zero-优化stage-123)
5. [流水线并行 (Pipeline Parallelism)](#5-流水线并行-pipeline-parallelism)
6. [张量并行 (Tensor Parallelism)](#6-张量并行-tensor-parallelism)
7. [序列并行 (Sequence Parallelism)](#7-序列并行-sequence-parallelism)
8. [3D 并行：组合策略](#8-3d-并行组合策略)
9. [实际案例分析](#9-实际案例分析)

---

## 1. 为什么需要多 GPU 并行

### 1.1 单 GPU 的极限

**计算极限**：世界最快超级计算机有 exaFLOPs 级算力，单 GPU 只有 ~1 PFLOP。

**内存极限**：模型越来越大，单个 GPU (80GB) 根本装不下：

```
70B 参数 × 16 bytes/param (AdamW) = 1.12 TB → 14 张 H100！
```

### 1.2 解决方案：多 GPU、多机器

```
节点内并行 (Intra-node):
  8 × GPU 通过 NVLink 高速互连 (900 GB/s on H100)

节点间并行 (Inter-node):
  多机器通过 InfiniBand/RoCE 互连 (较慢)
```

**目标**：
- 线性内存扩展：最大模型参数 ∝ GPU 数量
- 线性计算扩展：FLOPs ∝ GPU 数量

---

## 2. 集合通信原语

### 2.1 五大通信操作

| 操作 | 描述 | 通信量 |
|------|------|--------|
| **Broadcast** | 一个 → 所有 | P 份数据 |
| **Reduce** | 所有 → 一个 (求和) | 1 份数据 |
| **All-Reduce** | 所有 → 所有 (求和) | 等同 Reduce+Broadcast |
| **All-Gather** | 每人一片 → 每人全部 | P 份数据 |
| **Reduce-Scatter** | 每人全部 → 每人一片 (求和) | P 份数据 |

### 2.2 关键等价关系

```
All-Reduce = Reduce-Scatter + All-Gather
```

在带宽受限场景下，这种分解是最优的。这是 ZeRO 的理论基础。

### 2.3 GPU vs TPU 的网络差异

| | GPU | TPU |
|---|-----|-----|
| 拓扑 | All-to-All (最多 256) | 环面网格 (Toroidal Mesh) |
| 互连 | NVLink + InfiniBand | 自定义互连 |

---

## 3. 数据并行 (Data Parallelism)

### 3.1 朴素数据并行 (Naive DDP)

```
M 台机器，每台有完整的模型副本

1. 将 batch B 分成 M 份，每台处理 B/M
2. 每台独立做前向+反向
3. All-Reduce 汇总梯度
4. 每台用相同梯度更新参数 (保持同步)
```

**性能分析**：

| 维度 | 评价 |
|------|------|
| 计算扩展 | ✅ 每台处理 B/M (线性加速) |
| 通信开销 | 每步传输 2 × #params |
| 内存扩展 | ❌ 无！每台都存完整模型 |

### 3.2 内存问题详解

朴素 DDP 的内存布局（混合精度训练）：

```
每参数内存:
  BF16 模型参数:       2 bytes
  BF16 梯度:           2 bytes
  FP32 Master 权重:    4 bytes  ┐
  FP32 Adam 一阶矩:    4 bytes  ├── 优化器状态
  FP32 Adam 二阶矩:    4 bytes  ┘
  ─────────────────────────────
  总计:                16 bytes / param

→ 每台 GPU 都要存这 16 bytes/param！
```

---

## 4. ZeRO 优化：Stage 1/2/3

### 4.1 核心思想

> 将冗余的状态**分片 (shard)** 到不同 GPU 上，按需通信获取。

### 4.2 ZeRO Stage 1：优化器状态分片

```
之前: 每台 GPU 存全部优化器状态 (8 bytes/param × N_params)
之后: 每台只存 1/M 的优化器状态

流程:
1. 每台计算完整梯度 (在自己的 batch 子集上)
2. Reduce-Scatter 梯度 → 每台得到自己负责的梯度分片
3. 每台只更新自己负责的参数分片
4. All-Gather 参数 → 每台恢复完整参数
```

**通信量**: 2 × #params (与朴素 DDP 相同！)

**内存节省**: 优化器状态从 K 降到 K/N_gpu

### 4.3 ZeRO Stage 2：梯度也分片

```
在 Stage 1 基础上，梯度也不保留完整副本

流程:
1. 逐层反向传播
   1a. 每完成一层，立即 Reduce 该层梯度到负责的 GPU
   1b. 不再需要的梯度立即释放
2. 每台更新自己负责的参数
3. All-Gather 参数
```

**通信量**: 仍然 ~2 × #params

**内存**: 梯度也分片了 → 进一步节省

### 4.4 ZeRO Stage 3 (FSDP)：全部分片

```
参数、梯度、优化器状态全部分片

流程:
前向传播:
  - 需要某层参数时，All-Gather 获取
  - 用完立即释放
反向传播:
  - 需要参数时再次 All-Gather
  - 计算梯度后 Reduce-Scatter
  - 梯度和参数都立即释放
```

**关键优化**：**通信与计算重叠 (Overlap)**

```
时间线:
GPU 计算:  [前向层1] [前向层2] [前向层3] ...
通信:      [All-Gather层2] [All-Gather层3] ...
           ↑ 在计算层1的同时，预取层2的参数
```

**通信量**: 3 × #params (1.5× DDP)

**内存**: 真正的线性扩展！

### 4.5 对比总结

| | 通信原语 | 通信量 | 每参数内存 (BF16+FP32) |
|---|---------|--------|----------------------|
| **DDP** | 1× All-Reduce | 2×#params | 12 bytes |
| **ZeRO-1** | RS + AG | 2×#params | (4 + K/N) bytes |
| **ZeRO-2** | 增量RS + AG | ~2×#params | (2 + 10/N) bytes |
| **ZeRO-3 (FSDP)** | 2×AG + RS | 3×#params | 12/N bytes |

### 4.6 实际容量估算 (8×A100 80GB)

| 方案 | 最大模型参数 | 公式 (bytes/param) |
|------|-------------|-------------------|
| 基线 (无并行) | 6.7B | 12 |
| ZeRO-1 | 16B | 5 |
| ZeRO-2 | 24.6B | 2 + 10/8 |
| ZeRO-3 | 53.3B | 12/8 |

**ZeRO-1 是"免费"的——通信量不增加，但内存显著节省。应该总是使用。**

---

## 5. 流水线并行 (Pipeline Parallelism)

### 5.1 为什么需要模型并行

数据并行的两个根本限制：
1. **批量大小有上限**：batch_size 太大性能下降 (critical batch size)
2. **模型仍可能不适配**：ZeRO-1/2 不缩减参数内存，ZeRO-3 有延迟开销

### 5.2 朴素层级并行

```
将模型的不同层分配到不同 GPU:
GPU 0: 层 0-15
GPU 1: 层 16-31
GPU 2: 层 32-47
GPU 3: 层 48-63
```

**问题**：利用率极差！N 个 GPU 中每个只有 1/N 的时间在工作（等待前层完成）。

### 5.3 流水线并行

**核心改进**：将 batch 拆成多个 **micro-batch**，形成流水线。

```
时间→  Step1  Step2  Step3  Step4  Step5  Step6  Step7  Step8
GPU0:  F(μ1)  F(μ2)  F(μ3)  F(μ4)  B(μ4)  B(μ3)  B(μ2)  B(μ1)
GPU1:         F(μ1)  F(μ2)  F(μ3)  B(μ3)  B(μ2)  B(μ1)
GPU2:                F(μ1)  F(μ2)  B(μ2)  B(μ1)
GPU3:                       F(μ1)  B(μ1)

F = 前向, B = 反向, μ = micro-batch
```

**气泡时间比例**：

```
bubble_ratio = (n_stages - 1) / n_micro_batches
```

→ micro-batch 数量越多，气泡越小 → 需要大 batch size！

### 5.4 流水线并行的优缺点

| 优势 | 劣势 |
|------|------|
| 节省内存 (vs DDP) | 有"气泡"浪费 |
| 通信量小 (只传激活值 b×s×h) | 需要大 batch size |
| 点对点通信 (适合低带宽互连) | 实现复杂 |

### 5.5 高级流水线调度

**Zero-Bubble Pipeline**：
- 将反向传播拆成两部分：
  1. 反向传播激活梯度 (与下层有依赖)
  2. 计算权重梯度 (可以延迟)
- 权重梯度可以在任意时刻计算 → 填充气泡

---

## 6. 张量并行 (Tensor Parallelism)

### 6.1 核心思想

不是按层分割（深度），而是**按矩阵分割（宽度）**。

```
矩阵 Y = XA @ B

分割: A = [A1 | A2],  B = [B1]
                            [B2]

GPU 0: Y0 = X @ A1, 然后 Y0 @ B1
GPU 1: Y1 = X @ A2, 然后 Y1 @ B2

最后 All-Reduce: Y = Y0 + Y1
```

### 6.2 通信模式

```
前向传播: f = identity, g = All-Reduce
反向传播: f = All-Reduce, g = identity
```

**每个 Transformer 层需要 2 次 All-Reduce**（Attention 后一次，FFN 后一次）。

### 6.3 张量并行 vs 流水线并行

| | 张量并行 | 流水线并行 |
|---|---------|-----------|
| 气泡 | ❌ 无 | ✅ 有 |
| 通信量 | 8bsh × (N-1)/N per layer | bsh per micro-batch |
| 通信类型 | All-Reduce | 点对点 |
| Batch size 要求 | 无 | 大 |
| 适用场景 | **节点内** (高带宽) | **节点间** (低带宽) |
| 实现难度 | 简单 | 复杂 |

**核心规则**：**张量并行用在有高速互连的地方 (NVLink)，流水线并行用在慢速互连的地方 (InfiniBand)。**

---

## 7. 序列并行 (Sequence Parallelism)

### 7.1 动机：激活内存

张量并行和流水线并行能线性减少**参数内存**，但**激活内存**没有完全解决。

每层激活内存：
```
总激活 = 矩阵乘法相关 (可被 TP 分割) + 逐点操作 (LayerNorm, Dropout 等)
        ↑ 这部分已经分割了               ↑ 这 10sbh 项还没分割！
```

### 7.2 序列并行的方案

**观察**：LayerNorm、Dropout 等逐点操作是沿序列维度独立的。

```
将这些操作沿序列轴分片到不同 GPU:
GPU 0: 处理序列位置 0 ~ L/N
GPU 1: 处理序列位置 L/N ~ 2L/N
...

通信:
前向: g = All-Gather, ḡ = Reduce-Scatter
反向: 反过来
```

### 7.3 效果

结合张量并行 + 序列并行 → **激活内存也实现线性扩展**。

---

## 8. 3D 并行：组合策略

### 8.1 经验法则

```
1. 先让模型装得下:
   - 张量并行: 节点内 GPU (最多 8 个)
   - 流水线并行: 跨节点
   (或者用 ZeRO-3，取决于带宽)

2. 用剩余 GPU 做数据并行:
   - 用 ZeRO-1 (免费)
   - 如果 batch size 太小，用梯度累积

3. 如果 batch size 小:
   - 梯度累积 (gradient accumulation) 换取更好的通信效率
```

### 8.2 并行策略对比

| 策略 | 同步开销 | 内存扩展 | 带宽要求 | Batch size | 易用性 |
|------|---------|---------|---------|-----------|--------|
| DDP/ZeRO-1 | 每 batch | 无 | 2×#params | 线性 | ⭐⭐⭐⭐⭐ |
| FSDP (ZeRO-3) | 3×/FSDP block | 线性 | 3×#params | 线性 | ⭐⭐⭐⭐⭐ |
| Pipeline | 每 pipeline | 线性 | 激活值 | 线性 | ⭐⭐ |
| Tensor+Seq | 2×/层 | 线性 | 8×激活/层 | 无影响 | ⭐⭐⭐ |

### 8.3 Narayanan 2021 的 Scaling 策略

```
GPU 数量:    8    16    64    128   256   512   1024  2048  3072
TP 大小:     8     8     8     8     8     8     8     8     8
PP 大小:     1     2     8    16    32    32    24    15     9
DP 大小:     1     1     1     1     1     2     5    17    43

观察:
- TP 始终 = 8 (节点内所有 GPU)
- PP 先增大 (让模型装得下)
- 然后 DP 接管 (更多 GPU 时)
```

### 8.4 关键发现

- **TP = 8 通常最优**：即使有更多机器
- **激活重计算可以"回本"**：减少内存 → 允许更大 batch → 提高吞吐
- 仔细的 3D 并行可以实现**近线性扩展** (同 MFU, 更多 GPU)

---

## 9. 实际案例分析

### DeepSeek V3
- ZeRO Stage 1
- Pipeline Parallel (16-way)
- Expert Parallel (64-way, 8 nodes)
- Tensor + Sequence Parallel

### LLaMA 3 405B
- Stage 1: 小 batch 训练
- Stage 2: 主训练阶段
- Stage 3: 长上下文训练
- 大量 GPU 故障处理 (大规模训练的现实)

### Gemma 2
- ZeRO-3 + TP + SP + DP
- 适用于 2B, 9B, 27B 模型

### Yi
- ZeRO-1 + Tensor + Pipeline Parallel
- Yi-Lightning (2025): TP 被 Expert Parallel 替代

---

## 总结

### 核心要点

1. **没有单一解决方案**——需要组合多种并行策略
2. **简单的经验法则**足以指导大多数场景：TP(节点内) + PP(跨节点) + DP(剩余)
3. **ZeRO-1 是免费的**——应该总是使用
4. **通信与计算重叠**是关键——让 GPU 在等待数据时也在工作

### 并行策略选择速查

```
模型 < 8×GPU 内存? → 纯数据并行 (DDP + ZeRO-1)
模型 < 1节点内存?   → FSDP (ZeRO-3)
模型 > 1节点内存?   → TP(节点内) + PP(跨节点) + DP
需要长序列?         → + 序列并行
使用 MoE?          → + 专家并行
```

---

下一讲：Lecture 08 — 推理与量化
