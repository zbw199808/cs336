# Lecture 05 深入详细学习笔记：GPU 硬件与性能优化

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 5 - GPUs.pdf` (51 页幻灯片)

---

## 目录

1. [GPU 的历史与意义](#1-gpu-的历史与意义)
2. [GPU vs CPU：架构差异](#2-gpu-vs-cpu架构差异)
3. [GPU 执行模型](#3-gpu-执行模型)
4. [GPU 内存层次](#4-gpu-内存层次)
5. [Roofline 模型：性能分析框架](#5-roofline-模型性能分析框架)
6. [优化技巧 1：低精度计算](#6-优化技巧-1低精度计算)
7. [优化技巧 2：算子融合](#7-优化技巧-2算子融合)
8. [优化技巧 3：重计算](#8-优化技巧-3重计算)
9. [优化技巧 4：内存合并](#9-优化技巧-4内存合并)
10. [优化技巧 5：Tiling](#10-优化技巧-5tiling)
11. [矩阵乘法性能之谜](#11-矩阵乘法性能之谜)
12. [Flash Attention 解析](#12-flash-attention-解析)

---

## 1. GPU 的历史与意义

### 1.1 计算驱动可预测的性能提升

Kaplan et al 的 scaling laws 表明：更多计算 → 可预测的更好性能。

```
更快的硬件 + 更好的利用率 + 更好的并行化 = 性能提升
```

这意味着理解 GPU 不是可选的——它直接决定你的模型能有多好。

### 1.2 Dennard 缩放已终结

传统的 Dennard 缩放 (1980-2000s) 通过缩小晶体管来提速，但已经达到物理极限。

**替代方案**：通过大规模并行来持续提升。GPU 的并行能力在 10 年内提升了 **>1000 倍**。

> "没有 GPU 的缩放，就没有 LLM 的缩放。"

---

## 2. GPU vs CPU：架构差异

### 2.1 设计哲学对比

| | CPU | GPU |
|---|-----|-----|
| 优化目标 | **延迟** (每个线程快) | **吞吐量** (总处理量大) |
| 线程数 | 少量强力线程 | **大量轻量线程** |
| 计算单元 | 少量复杂 ALU | **大量简单 ALU** |
| 控制逻辑 | 大量 (分支预测等) | 很少 |
| 缓存 | 大 (低延迟访问) | 小 (更多空间给计算) |

### 2.2 为什么 GPU 适合深度学习？

深度学习的核心操作（矩阵乘法）本质上是**大规模并行**的——每个输出元素可以独立计算。GPU 的"很多弱线程"恰好匹配这种工作模式。

---

## 3. GPU 执行模型

### 3.1 三层结构

```
GPU
├── SM (Streaming Multiprocessor) × 108 (A100)
│   ├── SP (Streaming Processor) × many
│   │   └── 执行单个线程
│   └── Tensor Core × 若干
│       └── 矩阵乘法加速单元
├── L2 Cache (片上)
└── Global Memory / HBM (片外)
```

### 3.2 SIMT 执行模型

**SIMT = Single Instruction, Multiple Threads**

所有线程执行**相同的指令**，但操作**不同的数据**。

三个关键概念：

| 概念 | 定义 | 关键点 |
|------|------|--------|
| **Thread** | 最小执行单元 | 所有线程执行相同指令 |
| **Warp** | 32 个连续线程 | 线程以 warp 为单位同步执行 |
| **Block** | 一组线程 | 在一个 SM 上执行，共享 shared memory |

### 3.3 GPU vs TPU

| | GPU | TPU |
|---|-----|-----|
| 计算单元 | 多个 SM | 少量 TC (Tensor Core) |
| 矩阵乘法性能 | 类似 | 类似 |
| 编程模型 | Warp + Block | 只有 Block |
| 非矩阵操作 | 更灵活 | 较弱 |
| 组网方式 | NVLink/InfiniBand | 自定义互连 |

高层次上，GPU 和 TPU 的核心结构类似：轻量控制 + 快速大矩阵乘法单元 + 快速内存。

---

## 4. GPU 内存层次

### 4.1 层次结构

```
寄存器 (Register)          → 最快, 最小, 每线程私有
    ↓
共享内存 / L1 Cache (SRAM) → 快, ~192KB/SM, Block 内共享
    ↓
L2 Cache                    → 中等, ~40MB, 全 GPU 共享
    ↓
全局内存 / HBM (DRAM)       → 最慢, 80GB (A100/H100)
```

### 4.2 SRAM vs DRAM

| | SRAM (共享内存/L1) | DRAM (全局内存/HBM) |
|---|-------------------|-------------------|
| 速度 | **~8× 更快** | 基准 |
| 成本 | **~100× 更贵** | 基准 |
| 容量 | KB 级 | GB 级 |
| 访问方式 | Block 内直接访问 | 通过总线 |

### 4.3 核心矛盾

**计算能力的增长速度远超内存带宽**：

```
年份:  计算能力 ↑↑↑↑↑   内存带宽 ↑↑
→ 越来越难"喂饱"计算单元
→ 内存墙 (Memory Wall) 是主要瓶颈
```

### 4.4 线程与内存的关系

```
Thread → 可访问自己的 Register
       → 可访问 Block 的 Shared Memory
       → 可访问 Global Memory (慢！)

跨 Block 的数据共享 → 必须通过 Global Memory
```

---

## 5. Roofline 模型：性能分析框架

### 5.1 核心概念：算术强度

```
算术强度 (Arithmetic Intensity) = FLOPs / Bytes 移动
```

### 5.2 两种瓶颈

| 类型 | 条件 | 瓶颈 | 优化方向 |
|------|------|------|---------|
| **计算受限** (Compute-bound) | 高算术强度 | GPU 计算能力 | 更快的芯片 |
| **内存受限** (Memory-bound) | 低算术强度 | 内存带宽 | 减少数据移动 |

### 5.3 常见操作的分类

| 操作 | 算术强度 | 瓶颈类型 |
|------|---------|---------|
| 矩阵乘法 (大矩阵) | 高 | 计算受限 |
| 逐元素操作 (ReLU等) | ~1 | **内存受限** |
| 归一化 (LayerNorm) | 低 | **内存受限** |
| Softmax | 低 | **内存受限** |

**关键认知**：大多数非矩阵操作都是内存受限的。优化它们的关键不是减少 FLOPs，而是**减少数据移动**。

---

## 6. 优化技巧 1：低精度计算

### 6.1 更少的 bits = 更少的数据移动

```
例子: 逐元素 ReLU (x = max(0, x)) on 长度为 n 的向量

Float32:
  内存访问: 1 读 + 1 写 = 8 bytes/元素
  运算: 1 FLOP
  强度: 8 bytes/FLOP

Float16:
  内存访问: 1 读 + 1 写 = 4 bytes/元素
  运算: 1 FLOP
  强度: 4 bytes/FLOP  → 2× 改善！
```

### 6.2 Tensor Core 的低精度加速

Tensor Core 是 NVIDIA 从 Volta 架构引入的专用矩阵乘法电路：

```
矩阵乘法在 Tensor Core 上比普通 FP32 快 >10×！
```

现代 GPU 的 BF16 Tensor Core 峰值远高于 FP32 峰值，这就是为什么低精度训练如此重要。

---

## 7. 优化技巧 2：算子融合 (Operator Fusion)

### 7.1 类比

```
把 GPU 想象成工厂:
- 全局内存 = 仓库 (远)
- SRAM = 工厂车间 (近)
- 计算单元 = 工人

非融合: 仓库 → 工人做操作A → 仓库 → 工人做操作B → 仓库
融合:   仓库 → 工人连续做操作A和B → 仓库

减少了一半的"仓库搬运"！
```

### 7.2 具体例子

计算 `sin²(x) + cos²(x)`：

```
非融合 (5 个 CUDA kernel):
1. load x → compute sin(x) → store
2. load sin(x) → compute sin²(x) → store
3. load x → compute cos(x) → store
4. load cos(x) → compute cos²(x) → store
5. load sin²(x), cos²(x) → add → store

融合 (1 个 CUDA kernel):
1. load x → compute sin(x), cos(x), sin²(x), cos²(x), add → store
```

**5 次内存往返 → 1 次！**

### 7.3 自动融合

简单的逐元素融合可以由编译器自动完成：`torch.compile` 能自动发现并融合这类操作。

---

## 8. 优化技巧 3：重计算 (Recomputation)

### 8.1 反向传播的激活存储问题

反向传播需要前向传播中的激活值。标准做法是全部保存。

```
前向: x → σ(x) → σ(σ(x)) → σ(σ(σ(x)))
保存:     [a₁]    [a₂]      [a₃]

每步需要读写内存 → 3 层 sigmoid = 8 次内存访问
```

### 8.2 丢弃并重新计算

与其保存所有激活值，不如**丢弃**中间激活，在反向传播时**重新计算**：

```
前向: x → σ → σ → σ (不保存中间结果)

反向时: 从 x 重新计算 a₁ = σ(x), 再算 a₂ = σ(a₁)...

内存访问: 8 次 → 5 次 (减少 37.5%)
```

**关键洞察**：丢弃计算结果在内存受限的场景下可能是**最优的**——多做一些 FLOPs，但大幅减少内存访问。

这就是**梯度检查点 (Activation/Gradient Checkpointing)** 的原理。

---

## 9. 优化技巧 4：内存合并 (Memory Coalescing)

### 9.1 DRAM 的突发模式

DRAM 的读取不是按字节的，而是按**突发 (burst)** 的——每次读取会给你一整段连续内存。

```
一次 DRAM 读取 → 获得 128 字节的连续数据
```

### 9.2 合并访问 vs 非合并访问

```
合并 (Coalesced):
Warp 中 32 个线程访问连续内存地址 → 1 次 burst 搞定
[Thread 0 → addr 0] [Thread 1 → addr 1] ... [Thread 31 → addr 31]

非合并 (Non-coalesced):
Warp 中 32 个线程访问分散的内存地址 → 多次 burst
[Thread 0 → addr 0] [Thread 1 → addr 128] ... (随机)
```

### 9.3 矩阵乘法中的合并

对于行主序 (row-major) 矩阵：

```
沿行方向读取 → 内存连续 → 合并 ✅
沿列方向读取 → 内存不连续 → 非合并 ❌
```

这就是为什么矩阵存储顺序 (row-major vs column-major) 对性能影响很大。

---

## 10. 优化技巧 5：Tiling (分块)

### 10.1 核心思想

将大矩阵切成小块 (tiles)，加载到 Shared Memory 中，在 SRAM 上完成尽可能多的计算。

```
标准矩阵乘法:
C[i,j] = Σ_k A[i,k] × B[k,j]
每个输入元素从全局内存读取 N 次！

Tiled 矩阵乘法:
1. 加载 A 的一个 tile 和 B 的一个 tile 到 Shared Memory
2. 在 Shared Memory 中计算部分和
3. 加载下一对 tiles
4. ...

每个输入元素从全局内存只读 N/T 次！(T = tile 大小)
```

### 10.2 数学分析

```
非 Tiled: 每个元素读取 N 次 → 总全局内存访问 O(N³)
Tiled:    每个元素读取 N/T 次 → 总全局内存访问 O(N³/T)

→ 全局内存访问减少 T 倍！
```

### 10.3 Tiling 的复杂性

**Tile 大小不整除矩阵维度**：
- 边缘 tile 利用率低
- 需要 padding

**内存对齐**：
- Burst 模式要求内存地址对齐
- 如果 tile 边界不对齐 burst 边界，需要额外 padding

**影响 tile 大小的因素**：
- Shared Memory 大小限制
- 合并访问要求
- 矩阵维度的可整除性

---

## 11. 矩阵乘法性能之谜

### 11.1 为什么更大的矩阵更快？

直觉上，大矩阵需要更多计算，应该更慢。但实际 FLOP/s 随矩阵变大而**提高**。

### 11.2 两个关键因素

**因素 1: Tiling 对齐**

如果矩阵维度是 tile 大小的整数倍，tiling 效率最高。否则边缘 tile 浪费。

**因素 2: Wave 量化 (Wave Quantization)**

```
例子 (A100, 108 个 SM, tile 大小 256×128):

矩阵大小 1792:
  tiles = (1792/256) × (1792/128) = 7 × 14 = 98 tiles
  98 < 108 SMs → 一波搞定 ✅

矩阵大小 1793:
  tiles = (1793/256) × (1793/128) = 8 × 15 = 120 tiles
  120 > 108 SMs → 需要两波，第二波只有 12 tiles → 浪费 89% 的 SM！
```

这就是为什么矩阵乘法性能会呈现**周期性波动**——每当 tile 数刚好超过 SM 数的整数倍时，性能骤降。

**实践启示**：选择矩阵维度时，确保是常见 tile 大小 (64, 128, 256) 的整数倍！

---

## 12. Flash Attention 解析

### 12.1 Attention 的计算

```
标准 Attention:
S = Q × K^T           (矩阵乘法)
P = softmax(S)         (逐行 softmax)
O = P × V              (矩阵乘法)
```

### 12.2 标准实现的问题

```
标准流程:
1. Q, K, V 从 HBM 读入
2. 计算 S = Q × K^T → 写回 HBM (O(n²) 大小！)
3. 从 HBM 读 S → 计算 softmax → 写回 HBM
4. 从 HBM 读 P, V → 计算 O → 写回 HBM

S 矩阵的大小是 O(n²)！对于 n=4096, S 就是 4096×4096 = 64MB (float32)
→ 大量的 HBM 读写
```

### 12.3 Flash Attention 的解决方案

**核心技巧 1: Tiling**

不需要一次性计算完整的 S 矩阵！可以分块计算。

```
将 Q, K, V 分成 tiles:
Q = [Q₁, Q₂, ...]
K = [K₁, K₂, ...]
V = [V₁, V₂, ...]

对每对 (Qᵢ, Kⱼ):
  加载 tile 到 SRAM
  计算部分注意力
  写回部分结果
```

**核心技巧 2: 在线 Softmax**

标准 softmax 需要知道所有值才能计算 (需要全局 max)。Flash Attention 用**在线 softmax** (Milakov & Gimelshein 2018) 解决：

```
标准 softmax:
  1. 找到全局 max: m = max(x₁, ..., xₙ)
  2. 计算: softmax(xᵢ) = exp(xᵢ - m) / Σ exp(xⱼ - m)

在线 softmax:
  1. 逐 tile 处理，维护 running max 和 running sum
  2. 当看到新 tile 时，用"伸缩和" (telescoping sum) 修正之前的结果

  m_new = max(m_old, max(new_tile))
  correction = exp(m_old - m_new)
  sum_new = sum_old × correction + Σ exp(new_tile - m_new)
```

### 12.4 Flash Attention 的完整前向传播

结合 tiling + 在线 softmax + 融合指数运算：

```
for each Q tile:
    for each K, V tile:
        1. 加载 Qᵢ, Kⱼ, Vⱼ 到 SRAM
        2. 计算部分 Sᵢⱼ = Qᵢ × Kⱼ^T (在 SRAM 中)
        3. 融合: 计算 exp(Sᵢⱼ - running_max)
        4. 更新 running softmax (在线算法)
        5. 累积: Oᵢ += softmax_weights × Vⱼ
```

**反向传播**：不保存 S 矩阵，而是逐 tile 重计算（利用重计算技巧）。

### 12.5 Flash Attention 的效果

- **不改变输出结果**（精确等价！）
- 内存使用从 O(n²) 降到 O(n)
- 在长序列上速度提升 2-4×
- 已成为所有现代 LLM 训练的标配

---

## 总结：五大优化技巧

| 技巧 | 核心思想 | 适用场景 |
|------|---------|---------|
| **低精度** | 更少的 bits → 更少的数据移动 | 全局 |
| **算子融合** | 减少 HBM 读写次数 | 连续的逐元素操作 |
| **重计算** | 用计算换内存访问 | 反向传播中的激活值 |
| **内存合并** | 确保 warp 内连续访问 | 所有内存操作 |
| **Tiling** | 利用 SRAM 减少全局内存访问 | 矩阵乘法、Attention |

**核心认知**：

> 硬件驱动规模，低层细节决定什么能规模化、什么不能。
> 当前 GPU 的特性强烈鼓励围绕"矩阵乘法 + 数据移动"来思考。
> 仔细考虑 GPU 的特性 (合并、tiling、融合) 能带来显著的性能提升。

---

下一讲：Lecture 06 — GPU Kernel 编程 (Triton)
