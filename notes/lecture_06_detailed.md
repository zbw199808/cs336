# Lecture 06 深入详细学习笔记：GPU Kernel 编程

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_06.py` (可执行讲义)

---

## 目录

1. [概述：从理论到实践](#1-概述从理论到实践)
2. [Benchmarking：性能测量](#2-benchmarking性能测量)
3. [Profiling：性能分析](#3-profiling性能分析)
4. [Kernel Fusion 动机：GeLU 案例](#4-kernel-fusion-动机gelu-案例)
5. [CUDA Kernel 编写](#5-cuda-kernel-编写)
6. [Triton Kernel 编写](#6-triton-kernel-编写)
7. [torch.compile：自动编译](#7-torchcompile自动编译)
8. [Triton Softmax：聚合操作](#8-triton-softmax聚合操作)
9. [Triton 矩阵乘法与 Tiling](#9-triton-矩阵乘法与-tiling)

---

## 1. 概述：从理论到实践

本讲是 Lecture 05 (GPU 理论) 的实践延续：

```
Lecture 05: GPU 是什么、为什么快、性能瓶颈在哪
Lecture 06: 如何测量性能、如何写高效 GPU 代码
```

**5 种实现同一函数的方式**：

| 方式 | 工具 | 融合? | 易用性 |
|------|------|-------|--------|
| 手写 Python | PyTorch ops | ❌ 未融合 | ⭐⭐⭐⭐⭐ |
| PyTorch 内置 | C++/CUDA 预编译 | ✅ 已融合 | ⭐⭐⭐⭐⭐ |
| torch.compile | 自动 Triton 编译 | ✅ 自动融合 | ⭐⭐⭐⭐ |
| CUDA kernel | C++/CUDA 手写 | ✅ 手动融合 | ⭐⭐ |
| Triton kernel | Python DSL | ✅ 半自动融合 | ⭐⭐⭐ |

---

## 2. Benchmarking：性能测量

### 2.1 为什么必须 Benchmark

> "你可以读规格书和论文，但性能取决于你的库版本、硬件和工作负载……没有什么能替代实际测量。"

### 2.2 正确的 Benchmarking 方法

```python
def benchmark(run, num_warmups=1, num_trials=3):
    # 1. 预热 (Warmup): 首次运行可能因编译/缓存更慢
    for _ in range(num_warmups):
        run()

    # 2. 同步! GPU 操作是异步的
    torch.cuda.synchronize()

    # 3. 多次测量，取平均
    times = []
    for _ in range(num_trials):
        start = time.time()
        run()
        torch.cuda.synchronize()  # 等 GPU 完成！
        times.append(time.time() - start)

    return mean(times)
```

**关键细节**：
- `torch.cuda.synchronize()` 是必须的——GPU 操作是异步的，不同步会测到错误的时间
- 预热排除编译和缓存效应
- 多次运行捕获方差

### 2.3 Scaling 行为观察

对 MLP 模型的不同维度进行 scaling benchmark：
- **num_steps ×2**：时间近似 ×2（线性）
- **num_layers ×2**：时间近似 ×2（线性）
- **batch_size ×2**：时间增长 < ×2（GPU 并行有余量）
- **dim ×2**：时间增长 > ×2（矩阵乘法 FLOPs 是 O(d²)）

> "时间并不总是可预测的——CUDA kernel、硬件的非均匀性导致了性能谜题。"

---

## 3. Profiling：性能分析

### 3.1 Benchmarking vs Profiling

| | Benchmarking | Profiling |
|---|-------------|-----------|
| 回答 | "花了多长时间？" | "时间花在哪里？" |
| 粒度 | 端到端 | 每个 kernel/操作 |
| 用途 | 比较实现 | 定位瓶颈 |

### 3.2 PyTorch Profiler 使用

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    run()
    torch.cuda.synchronize()

# 打印耗时最长的操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 3.3 从 Profile 中读到什么

**CUDA Kernel 名称包含实现信息**：

```
cutlass_80_simt_sgemm_256x128_8x4_nn_align1
│        │    │      │         │
│        │    │      │         └── 内存对齐
│        │    │      └── tile 大小 256×128
│        │    └── 单精度矩阵乘法
│        └── GPU 架构 (sm80 = A100)
└── NVIDIA 线性代数库
```

**不同矩阵维度会调用不同的 CUDA kernel**——库会根据问题大小自动选择最优实现。

---

## 4. Kernel Fusion 动机：GeLU 案例

### 4.1 GeLU 的两种实现

**手写 Python (未融合)**：
```python
def manual_gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))
    # 每个中间操作都是独立 kernel → 每步读写 HBM
```

**PyTorch 内置 (已融合)**：
```python
def pytorch_gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")
    # 一个融合 kernel → 只读写一次 HBM
```

### 4.2 性能对比

Profile 显示：
- `manual_gelu`：多个 CUDA kernel 串行（多次 HBM 读写）
- `pytorch_gelu`：**只有 1 个 CUDA kernel**

性能差异可达数倍！这就是 kernel fusion 的威力。

---

## 5. CUDA Kernel 编写

### 5.1 CUDA 编程模型

```
简化图景: 写 f(i), CUDA 对所有 i = 0, ..., N-1 并行执行 f(i)

线程定位:
  blockIdx  = 当前线程块在 Grid 中的位置
  blockDim  = 每个线程块的大小
  threadIdx = 当前线程在块中的位置
  全局 index = blockIdx * blockDim + threadIdx
```

### 5.2 GeLU CUDA Kernel (概念)

```c
__global__ void gelu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}
```

**性能结果**：
- 比手写 Python 快（因为融合了）
- 但比 PyTorch 内置慢（PyTorch 有更多优化）

### 5.3 CUDA 的局限

逐元素操作在 CUDA 中很简单。但更复杂的操作（matmul, softmax, RMSNorm）需要手动管理共享内存、同步等——非常复杂。

---

## 6. Triton Kernel 编写

### 6.1 Triton 简介

OpenAI 2021 年开发，目标是让 GPU 编程更可及：

```
                          CUDA        Triton
内存合并 (Coalescing)     手动         自动
共享内存管理               手动         自动
SM 内调度                 手动         自动
SM 间调度                 手动         手动
```

**核心区别**：CUDA 以**线程**为单位思考，Triton 以**线程块**为单位思考。

### 6.2 Triton GeLU Kernel

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # 确定当前 block 负责的数据范围
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 边界处理
    mask = offsets < num_elements

    # 从全局内存读取
    x = tl.load(x_ptr + offsets, mask=mask)

    # 计算 GeLU (在 SRAM 中完成！)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # 写回全局内存
    tl.store(y_ptr + offsets, y, mask=mask)
```

**关键要素**：
- `tl.program_id(0)` — 当前 block 的 ID
- `tl.arange(0, BLOCK_SIZE)` — block 内的偏移量
- `tl.load` / `tl.store` — 从/到全局内存
- 所有中间计算在寄存器/SRAM 中完成（自动管理）

### 6.3 PTX 汇编查看

Triton 编译后生成 PTX (Parallel Thread Execution) 代码，可以查看：

```
ld.global.*  — 从全局内存读取
st.global.*  — 写入全局内存
%ctaid.x     — block index
%tid.x       — thread index
%f*          — 浮点寄存器
%r*          — 整数寄存器
```

**有趣发现**：Triton 自动进行了**线程粗化 (thread coarsening)**——一个线程处理 8 个元素。

---

## 7. torch.compile：自动编译

### 7.1 最简单的优化方式

```python
compiled_gelu = torch.compile(manual_gelu)
# 自动将 Python 代码编译为优化的 Triton kernel
```

### 7.2 性能对比 (GeLU, dim=16384)

```
manual_gelu    → 最慢 (多个未融合 kernel)
cuda_gelu      → 中等 (手写融合)
triton_gelu    → 快 (Triton 自动优化)
compiled_gelu  → 快 (torch.compile 自动融合)
pytorch_gelu   → 最快 (高度优化的内置实现)
```

> "自动编译器 (Triton, torch.compile) 会随时间变得越来越好。"

---

## 8. Triton Softmax：聚合操作

### 8.1 从逐元素到行级聚合

GeLU 是逐元素的（每个元素独立），softmax 需要**整行聚合**（需要行最大值和行求和）。

### 8.2 手写 Softmax 的内存分析

```python
def manual_softmax(x):
    M, N = x.shape
    x_max = x.max(dim=1)[0]          # MN reads, M writes
    x = x - x_max[:, None]            # MN+M reads, MN writes
    numerator = torch.exp(x)          # MN reads, MN writes
    denominator = numerator.sum(dim=1) # MN reads, M writes
    y = numerator / denominator[:, None] # MN+M reads, MN writes

    # 总计: 5MN + 2M reads, 3MN + 2M writes
    # 理论最优: MN reads, MN writes → 约 4× 改善空间！
```

### 8.3 Triton Softmax Kernel

**核心设计**：每个 block 处理矩阵的一行。

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride,
                          num_cols, BLOCK_SIZE: tl.constexpr):
    # 每个 block 处理一行
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 一次性读取整行到 SRAM
    x_row = tl.load(x_ptr + row_idx * x_row_stride + col_offsets,
                    mask=col_offsets < num_cols, other=float("-inf"))

    # 全部在 SRAM 中计算 (无 HBM 读写！)
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 一次性写回
    tl.store(y_ptr + row_idx * y_row_stride + col_offsets,
             y_row, mask=col_offsets < num_cols)
```

**从 5MN → MN 的内存访问**——这就是融合的力量。

---

## 9. Triton 矩阵乘法与 Tiling

### 9.1 Tiling 策略

```
标准 (naive): 每个元素需要从 DRAM 读 MKN 次
Tiled:       分块加载到共享内存
  - 加载 A 和 B 的 tile 到 SRAM
  - 在 SRAM 中做小矩阵乘法
  - 写回部分和
```

### 9.2 L2 Cache 利用

计算 9 个输出元素时，block 处理顺序影响 L2 cache 命中率：

```
行主序遍历: 加载 9 + 81 = 90 个 block (每行的 B 都要重新加载)
分组遍历:   加载 27 + 27 = 54 个 block (B 的 block 在 L2 中可复用)
```

### 9.3 为什么自己写 matmul kernel？

通常不需要——cuBLAS/CUTLASS 已经极度优化。但如果需要**融合 matmul 和后续操作**（如 `gelu(A @ B)`），自写 kernel 可以避免中间结果的 HBM 读写。

---

## 总结

### 核心原则

> **组织计算以最小化读写** (Organize computation to minimize reads/writes)

### 关键工具链

```
理解性能:  Benchmarking → Profiling → PTX 分析
优化性能:  torch.compile (最简单) → Triton (灵活) → CUDA (极致控制)
```

### 关键概念

| 概念 | 说明 |
|------|------|
| Kernel Fusion | 合并操作减少 HBM 读写 |
| Tiling | 利用 SRAM 减少全局内存访问 |
| 编程模型差距 | PyTorch/Triton/PTX 和实际硬件之间有差距 → 性能谜题 |

---

下一讲：Lecture 07 — 并行策略基础
