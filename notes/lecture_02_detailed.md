# Lecture 02 深入详细学习笔记：PyTorch 基础与资源核算

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_02.py` (可执行讲义)

---

## 目录

1. [概述与激励问题](#1-概述与激励问题)
2. [张量基础与内存核算](#2-张量基础与内存核算)
3. [浮点数据类型深入对比](#3-浮点数据类型深入对比)
4. [GPU 上的张量](#4-gpu-上的张量)
5. [张量操作详解](#5-张量操作详解)
6. [Einops：优雅的张量操作](#6-einops优雅的张量操作)
7. [计算量 (FLOPs) 核算](#7-计算量-flops-核算)
8. [梯度计算与反向传播 FLOPs](#8-梯度计算与反向传播-flops)
9. [模型构建：从参数到模块](#9-模型构建从参数到模块)
10. [训练实践：数据加载与随机性](#10-训练实践数据加载与随机性)
11. [优化器演进：从 SGD 到 Adam](#11-优化器演进从-sgd-到-adam)
12. [训练循环与检查点](#12-训练循环与检查点)
13. [混合精度训练](#13-混合精度训练)
14. [核心公式总结](#14-核心公式总结)

---

## 1. 概述与激励问题

### 1.1 本讲定位

本讲从底向上，构建训练模型所需的所有**原语** (primitives)：

```
张量 (Tensors) → 模型 (Models) → 优化器 (Optimizers) → 训练循环 (Training Loop)
```

贯穿始终关注两类资源的核算：
- **内存 (Memory)**：以 GB 为单位
- **计算 (Compute)**：以 FLOPs 为单位

### 1.2 "信封背面"估算

这是 LLM 工程师最核心的技能之一——快速估算资源需求。

#### 问题 1：训练 70B 模型需要多久？

> 在 1024 张 H100 上训练 70B 参数模型，处理 15T tokens，需要多少天？

```python
# 总计算量公式：6 × N × D (N=参数量, D=token数)
total_flops = 6 * 70e9 * 15e12  # = 6.3e24 FLOPs

# H100 理论算力 (BF16, 非稀疏)
h100_flop_per_sec = 1979e12 / 2  # ≈ 989.5 TFLOP/s

# 假设 MFU (Model FLOPs Utilization) = 50%
mfu = 0.5

# 每天实际算力
flops_per_day = h100_flop_per_sec * mfu * 1024 * 86400
# = 989.5e12 * 0.5 * 1024 * 86400
# ≈ 4.38e22 FLOPs/天

# 所需天数
days = total_flops / flops_per_day  # ≈ 144 天
```

**关键洞察**：
- `6 × N × D` 是训练 FLOPs 的核心公式（前向 2ND + 反向 4ND）
- MFU 50% 是实际中相当好的利用率
- 即使有 1024 张顶级 GPU，训练 70B 模型仍需约 5 个月

#### 问题 2：8 张 H100 能训练多大的模型？

```python
h100_memory = 80e9  # 80 GB 每张
total_memory = h100_memory * 8  # = 640 GB

# AdamW 每个参数的内存开销 (float32):
# - 参数本身:      4 bytes
# - 梯度:          4 bytes
# - 一阶矩 (m):    4 bytes (Adam optimizer state)
# - 二阶矩 (v):    4 bytes (Adam optimizer state)
bytes_per_parameter = 4 + 4 + (4 + 4)  # = 16 bytes

max_parameters = total_memory / bytes_per_parameter  # = 40B 参数
```

**重要注意**：
- **Caveat 1**：如果用 BF16 存储参数和梯度 (2+2)，加一份 FP32 参数副本 (4)，总共仍是 16 bytes——不节省内存，但计算更快
- **Caveat 2**：这个估算**没有包含激活值 (activations)**，激活值取决于 batch size 和 sequence length，是另一个显著的内存开销

---

## 2. 张量基础与内存核算

### 2.1 张量是什么

张量是存储一切的基础数据结构：参数、梯度、优化器状态、数据、激活值。

```python
# 创建方式
x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从数据创建
x = torch.zeros(4, 8)                       # 全零
x = torch.ones(4, 8)                        # 全一
x = torch.randn(4, 8)                       # 标准正态分布采样
x = torch.empty(4, 8)                       # 分配但不初始化 (更快)
```

**为什么用 `torch.empty`？** 当你后续要用自定义逻辑初始化时（如截断正态分布），先分配再初始化比直接创建更灵活：

```python
x = torch.empty(4, 8)
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断在 [-2, 2]
```

### 2.2 内存计算

内存 = **元素数量** × **每个元素的字节数**

```python
x = torch.zeros(4, 8)          # float32 by default
x.numel()                      # 32 个元素
x.element_size()               # 4 bytes/element (float32)
memory = x.numel() * x.element_size()  # = 128 bytes
```

**实际规模感**：GPT-3 的一个前馈层矩阵 (12288×4 × 12288)：

```python
memory = 12288 * 4 * 12288 * 4  # = 2.3 GB (float32)
```

一个矩阵就 2.3 GB！GPT-3 有 96 层，每层有多个这样的矩阵。

---

## 3. 浮点数据类型深入对比

### 3.1 float32 (单精度)

```
格式: 1 位符号 + 8 位指数 + 23 位尾数 = 32 bits = 4 bytes
```

- 科学计算的基线，深度学习的传统默认类型
- 动态范围：约 ±3.4 × 10^38
- 精度：约 7 位有效数字

### 3.2 float16 (半精度)

```
格式: 1 位符号 + 5 位指数 + 10 位尾数 = 16 bits = 2 bytes
```

- 内存减半，计算速度提升
- **致命缺陷**：动态范围有限，小数容易下溢 (underflow)

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0  # 下溢为 0！
```

训练中如果梯度下溢为 0，模型参数将无法更新，导致训练不稳定甚至崩溃。

### 3.3 bfloat16 (Brain Float)

```
格式: 1 位符号 + 8 位指数 + 7 位尾数 = 16 bits = 2 bytes
```

Google Brain 在 2018 年专门为深度学习设计：
- **关键创新**：保留与 float32 **相同的 8 位指数**，牺牲尾数精度
- 与 float16 相同的内存和计算效率
- 与 float32 相同的动态范围！

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0  # 不会下溢！
```

**为什么精度损失可以接受？** 深度学习对精确值的要求远低于对数值范围的要求。参数值的微小变化（精度损失）对模型几乎没有影响，但下溢到 0（范围不足）会直接破坏训练。

### 3.4 FP8 (8位浮点)

2022 年标准化，专为机器学习工作负载设计。H100 支持两个变体：

| 变体 | 格式 | 范围 | 用途 |
|------|------|------|------|
| E4M3 | 1+4+3 = 8 bits | [-448, 448] | 前向传播 (需要精度) |
| E5M2 | 1+5+2 = 8 bits | [-57344, 57344] | 反向传播 (需要范围) |

### 3.5 数据类型对比总结

| 类型 | 字节数 | 动态范围 | 精度 | 训练稳定性 | 计算速度 |
|------|--------|----------|------|------------|----------|
| float32 | 4 | ✅ 极好 | ✅ 极好 | ✅ 稳定 | ❌ 最慢 |
| float16 | 2 | ❌ 有限 | ⚠️ 中等 | ⚠️ 有下溢风险 | ✅ 快 |
| bfloat16 | 2 | ✅ 同 fp32 | ⚠️ 较低 | ✅ 比 fp16 稳定 | ✅ 快 |
| fp8 | 1 | ⚠️ 有限 | ❌ 很低 | ⚠️ 需要特殊处理 | ✅✅ 最快 |

---

## 4. GPU 上的张量

### 4.1 CPU vs GPU

默认情况下，张量存储在 CPU 内存中。要利用 GPU 的大规模并行性，必须将数据移到 GPU 内存。

```
CPU 内存 (系统 RAM)          GPU 内存 (HBM)
┌─────────────┐            ┌─────────────┐
│  较大 (TB级)  │ ──PCIe──→ │  较小 (80GB) │
│  通用计算     │            │  大规模并行   │
└─────────────┘            └─────────────┘
```

### 4.2 张量在 GPU 间的移动

```python
# 检查 GPU 可用性
torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

# CPU → GPU
x = torch.zeros(32, 32)           # 在 CPU 上
y = x.to("cuda:0")                 # 复制到 GPU 0
assert y.device == torch.device("cuda", 0)

# 直接在 GPU 上创建
z = torch.zeros(32, 32, device="cuda:0")  # 直接在 GPU 内存中分配
```

### 4.3 GPU 内存追踪

```python
before = torch.cuda.memory_allocated()  # 当前 GPU 内存占用
y = x.to("cuda:0")
z = torch.zeros(32, 32, device="cuda:0")
after = torch.cuda.memory_allocated()
memory_used = after - before  # = 2 × (32 × 32 × 4) = 8192 bytes
```

---

## 5. 张量操作详解

每个操作都有内存和计算代价。理解这些代价是资源核算的基础。

### 5.1 张量存储模型 (Storage)

PyTorch 的张量**不是**独立的数据块，而是指向连续内存的**指针 + 元数据**。

```
逻辑视图:                     底层存储 (连续内存):
┌───┬───┬───┬───┐            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
│ 0 │ 1 │ 2 │ 3 │
├───┼───┼───┼───┤            stride(0) = 4  (下一行跳 4 个元素)
│ 4 │ 5 │ 6 │ 7 │            stride(1) = 1  (下一列跳 1 个元素)
├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │            element[r,c] = storage[r × stride(0) + c × stride(1)]
├───┼───┼───┼───┤            element[1,2] = storage[1 × 4 + 2 × 1] = storage[6] = 6
│12 │13 │14 │15 │
└───┴───┴───┴───┘
```

**为什么这个设计很重要？** 它使得许多操作可以**零拷贝**完成——只需改变元数据 (stride, offset)。

### 5.2 视图 (View) 操作——零拷贝

这些操作不会创建新的内存分配，只是创建对同一底层存储的不同"视角"：

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

# 取一行 → 零拷贝
y = x[0]              # → tensor([1., 2., 3.])
assert same_storage(x, y)  # 共享底层存储！

# 取一列 → 零拷贝
y = x[:, 1]           # → tensor([2., 5.])
assert same_storage(x, y)

# Reshape → 零拷贝
y = x.view(3, 2)      # 2×3 → 3×2
assert same_storage(x, y)

# 转置 → 零拷贝
y = x.transpose(1, 0)  # 交换行列
assert same_storage(x, y)
```

**危险：共享存储意味着修改会互相影响！**

```python
x[0][0] = 100
assert y[0][0] == 100  # y 也被改了！
```

### 5.3 Contiguous 与何时需要拷贝

转置后的张量虽然共享存储，但元素在内存中不再连续 (non-contiguous)：

```python
y = x.transpose(1, 0)
assert not y.is_contiguous()

# 无法对 non-contiguous 张量做 view
y.view(2, 3)  # RuntimeError!

# 解决方案：先调用 .contiguous() 创建一份连续的拷贝
y = x.transpose(1, 0).contiguous().view(2, 3)
assert not same_storage(x, y)  # 现在是独立的存储了
```

**总结**：
- **View 操作**：免费 (零内存、零计算)
- **Copy 操作**：需要额外内存和计算

### 5.4 逐元素操作 (Elementwise)

对张量每个元素独立应用操作，返回同形状的新张量：

```python
x = torch.tensor([1, 4, 9])
x.pow(2)    # → [1, 16, 81]    每个元素平方
x.sqrt()    # → [1, 2, 3]       每个元素开方
x.rsqrt()   # → [1, 0.5, 0.33]  每个元素 1/sqrt(x)

x + x       # → [2, 8, 18]      逐元素加
x * 2       # → [2, 8, 18]      标量乘
```

**特别重要：上三角矩阵 `triu`**

```python
mask = torch.ones(3, 3).triu()
# tensor([[1, 1, 1],
#         [0, 1, 1],
#         [0, 0, 1]])
```

这用于构建 **因果注意力掩码 (Causal Attention Mask)**——确保位置 i 只能关注位置 ≤ i 的 token。

### 5.5 矩阵乘法——深度学习的核心

```python
x = torch.ones(16, 32)   # (B, D)
w = torch.ones(32, 2)    # (D, K)
y = x @ w                # (B, K) → size [16, 2]
```

**批量矩阵乘法**：实际训练中有 batch 维度和 sequence 维度：

```python
x = torch.ones(4, 8, 16, 32)  # (batch, seq, tokens, D)
w = torch.ones(32, 2)          # (D, K)
y = x @ w                      # (4, 8, 16, 2)
# PyTorch 自动在前面的维度上广播
```

---

## 6. Einops：优雅的张量操作

### 6.1 动机：传统代码的可读性问题

```python
# 传统 PyTorch
z = x @ y.transpose(-2, -1)  # -2, -1 是什么意思？容易搞混
```

维度用数字编号时，代码难以理解，且容易出错。

### 6.2 jaxtyping：类型注解维度

```python
# 旧方式
x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden (只能靠注释)

# 新方式：用 jaxtyping 注解
x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
```

**注意**：这只是文档，不做运行时强制检查。但极大提升了代码可读性。

### 6.3 einsum：广义矩阵乘法

Einstein summation notation——用命名维度进行张量收缩：

```python
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)

# 旧方式 (难以理解)
z = x @ y.transpose(-2, -1)

# 新方式 (清晰明了)
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

**规则**：输出中没有出现的维度会被求和 (sum over)。这正是矩阵乘法的本质——对"hidden"维度求和。

可用 `...` 表示任意数量的广播维度：
```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

### 6.4 reduce：命名维度上的归约

```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# 旧方式
y = x.sum(dim=-1)  # dim=-1 是哪个？

# 新方式
y = reduce(x, "... hidden -> ...", "sum")  # 明确对 hidden 求和
```

### 6.5 rearrange：维度的分拆与合并

这在多头注意力 (Multi-Head Attention) 中非常常见：

```python
# 把 total_hidden 拆分为 heads × hidden1
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
# 现在 x 的形状: (2, 3, 2, 4)

# 对每个 head 独立做线性变换
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# 合并回去
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
# 现在 x 的形状: (2, 3, 8)
```

---

## 7. 计算量 (FLOPs) 核算

### 7.1 基本概念

**两个容易混淆的缩写**（发音相同！）：

| 缩写 | 含义 | 性质 |
|------|------|------|
| **FLOPs** | Floating-point operations | 计算量 (一个数字) |
| **FLOP/s** | Floating-point operations per second | 算力 (速度) |

### 7.2 直觉建立

| 事件 | FLOPs |
|------|-------|
| 训练 GPT-3 (2020) | 3.14 × 10^23 |
| 训练 GPT-4 (2023, 推测) | 2 × 10^25 |
| 美国行政令报告阈值 | ≥ 10^26 (2025年被撤销) |

### 7.3 GPU 算力规格

| GPU | float32 | bfloat16 (Tensor Core) |
|-----|---------|------------------------|
| A100 | 19.5 TFLOP/s | 312 TFLOP/s |
| H100 | 67.5 TFLOP/s | ~990 TFLOP/s (非稀疏) |

**注意**：BF16 的理论算力比 FP32 高 **16 倍** (A100) 到 **15 倍** (H100)！这就是为什么用低精度训练如此重要。

### 7.4 矩阵乘法的 FLOPs

对于 (B×D) @ (D×K) 的矩阵乘法：

```
FLOPs = 2 × B × D × K
```

为什么是 2？对于输出中的每个元素 y[i,k]：
- 需要 D 次**乘法** (x[i,j] × w[j,k])
- 需要 D-1 次**加法** (累加)
- 约为 2D 次浮点运算
- 总共 B × K 个输出元素 → 2 × B × D × K

### 7.5 其他操作的 FLOPs

- **逐元素操作**：m×n 矩阵 → O(mn) FLOPs
- **矩阵加法**：m×n 矩阵 → mn FLOPs

**关键结论**：对于足够大的矩阵，**矩阵乘法主导所有计算**。其他操作的 FLOPs 相比之下可以忽略不计。

### 7.6 从 FLOPs 推广到 Transformer

```
前向传播 FLOPs ≈ 2 × (token 数) × (参数量)
```

这个简洁的公式对 Transformer 也近似成立（一阶近似），因为 Transformer 的计算主要是矩阵乘法。

### 7.7 Model FLOPs Utilization (MFU)

**定义**：

```
MFU = 实际 FLOP/s / 理论峰值 FLOP/s
```

这是衡量你"榨取"GPU 效率的核心指标：

```python
actual_num_flops = 2 * B * D * K
actual_time = time_matmul(x, w)           # 实际耗时
actual_flop_per_sec = actual_num_flops / actual_time
mfu = actual_flop_per_sec / promised_flop_per_sec
```

| MFU 范围 | 评价 |
|----------|------|
| ≥ 50% | 很好 (矩阵乘法主导时可达到) |
| 30-50% | 一般 (通信/内存开销较大) |
| < 30% | 需要优化 |

**实际案例**：
- PaLM (Google, 540B): 46.2% MFU on 6144 TPUv4
- MegaScale (ByteDance, 175B): 55.2% MFU on 12288 GPUs
- Megatron-LM: 52% MFU on 3072 GPUs for 1T model

---

## 8. 梯度计算与反向传播 FLOPs

### 8.1 基础梯度计算

```python
# 简单线性模型: y = x · w, loss = 0.5 * (y - 5)²
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)

pred_y = x @ w          # 前向
loss = 0.5 * (pred_y - 5).pow(2)
loss.backward()          # 反向

# 结果
w.grad  # → tensor([1., 2., 3.])  = x * (pred_y - 5)
```

**重要细节**：
- 只有 `requires_grad=True` 的张量才会计算梯度
- `loss.grad` 和 `x.grad` 都是 `None`（不需要计算）
- 梯度计算用**链式法则 (Chain Rule)**

### 8.2 两层线性模型的 FLOPs 分析

模型结构：`x --w1--> h1 --w2--> h2 -> loss`

**前向传播**：
```
h1 = x @ w1        → 2 × B × D × D FLOPs
h2 = h1 @ w2       → 2 × B × D × K FLOPs
前向总计: 2B(D² + DK)
```

**反向传播**（聚焦 w2 层）：

链式法则展开：

```
# 计算 w2 的梯度 (参数梯度)
w2.grad[j,k] = Σ_i h1[i,j] * h2.grad[i,k]
→ 本质是 h1.T @ h2.grad  → 2 × B × D × K FLOPs

# 计算 h1 的梯度 (传递给上一层)
h1.grad[i,j] = Σ_k w2[j,k] * h2.grad[i,k]
→ 本质是 h2.grad @ w2.T  → 2 × B × D × K FLOPs
```

w2 层的反向传播需要 **2 次矩阵乘法**：一次算参数梯度，一次传梯度到上层。

类似地，w1 层也需要 4 × B × D × D FLOPs。

### 8.3 核心结论：6ND 法则

```
╔═══════════════════════════════════════════════════════════╗
║  前向传播:  2 × (token数) × (参数量) FLOPs               ║
║  反向传播:  4 × (token数) × (参数量) FLOPs               ║
║  训练总计:  6 × (token数) × (参数量) FLOPs               ║
╚═══════════════════════════════════════════════════════════╝
```

**为什么反向是前向的 2 倍？** 每一层在反向传播时需要做 2 次矩阵乘法（算参数梯度 + 传递梯度），而前向只做 1 次。

这就是开头 "信封背面估算" 中 `6 × 70e9 × 15e12` 的来源。

---

## 9. 模型构建：从参数到模块

### 9.1 nn.Parameter

```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
# nn.Parameter 本质是 torch.Tensor + requires_grad=True
# 注册到 nn.Module 后可以被 optimizer 自动发现
```

### 9.2 参数初始化的重要性

**问题**：如果参数用标准正态分布初始化，输出的方差会随输入维度线性增长：

```python
x = torch.randn(input_dim)  # 各元素 ~ N(0, 1)
w = torch.randn(input_dim, output_dim)  # 各元素 ~ N(0, 1)
output = x @ w
# output[j] = Σ_i x[i] * w[i,j]
# Var(output[j]) = input_dim × Var(x[i]) × Var(w[i,j]) = input_dim
# 即 output 的标准差 ∝ sqrt(input_dim)
```

当 `input_dim = 16384` 时，输出值约在 ±128 的量级——这会导致梯度爆炸。

**解决方案：Xavier 初始化**

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
# 现在 Var(output[j]) = 1，与 input_dim 无关
```

**更安全的版本：截断正态分布**

```python
w = nn.Parameter(nn.init.trunc_normal_(
    torch.empty(input_dim, output_dim),
    std=1/np.sqrt(input_dim), a=-3, b=3
))
# 截断到 [-3σ, 3σ]，避免极端异常值
```

### 9.3 构建自定义模型

课程演示了一个简单的深度线性模型：

```python
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim) / np.sqrt(input_dim)
        )

    def forward(self, x):
        return x @ self.weight


class Cruncher(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim) for _ in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x.squeeze(-1)  # 移除最后的维度 1
```

**参数审计**：

```python
model = Cruncher(dim=64, num_layers=2)
# 参数量:
# layers.0.weight: 64 × 64 = 4096
# layers.1.weight: 64 × 64 = 4096
# final.weight:    64 × 1  = 64
# 总计: 8256 个参数
```

**重要**：创建模型后，必须移到 GPU 上！

```python
model = model.to(device)  # 所有参数一次性移到 GPU
```

---

## 10. 训练实践：数据加载与随机性

### 10.1 随机性管理

随机性出现在：参数初始化、dropout、数据顺序、采样等。

**最佳实践**：为每个随机性来源传入不同的随机种子，且设置所有三个随机数生成器：

```python
seed = 0
torch.manual_seed(seed)      # PyTorch
np.random.seed(seed)          # NumPy
random.seed(seed)             # Python 标准库
```

**为什么重要？** 确定性 (determinism) 对调试至关重要——只有能重现 bug，才能定位和修复它。

### 10.2 数据序列化与内存映射

语言模型的训练数据是分词器输出的整数序列：

```python
# 序列化
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
data.tofile("data.npy")

# 反序列化 — 使用 memmap！
data = np.memmap("data.npy", dtype=np.int32)
```

**为什么用 `memmap` 而不是 `np.load`？**
- LLaMA 的训练数据是 **2.8TB**
- 全部加载到内存是不可能的
- `memmap` 利用操作系统的虚拟内存，只在访问时才从磁盘加载对应的页

### 10.3 Batch 采样

```python
def get_batch(data, batch_size, sequence_length, device):
    # 随机采样 batch_size 个起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))

    # 从数据中提取序列
    x = torch.tensor([
        data[start:start + sequence_length]
        for start in start_indices
    ])  # shape: (batch_size, sequence_length)

    return x
```

### 10.4 Pinned Memory 优化

```python
# 默认: CPU 张量在分页内存中
x = torch.tensor(...)

# 优化: 固定到物理内存
x = x.pin_memory()

# 异步传输到 GPU (不阻塞 CPU)
x = x.to(device, non_blocking=True)
```

**好处**：可以在 GPU 处理当前 batch 的同时，CPU 异步加载下一个 batch。

```
时间线:
CPU: [加载 batch 1] [加载 batch 2] [加载 batch 3] ...
GPU:                [处理 batch 1] [处理 batch 2] ...
                    ↑ 重叠执行，无等待
```

---

## 11. 优化器演进：从 SGD 到 Adam

### 11.1 优化器家族谱系

```
SGD (随机梯度下降)
  │
  ├── + 动量 (exponential averaging of grad) → Momentum
  │
  ├── + 按 grad² 自适应学习率 → AdaGrad
  │                                    │
  │                                    └── + 指数平均 grad² → RMSProp
  │                                                             │
  └── Momentum + RMSProp ──────────────────────────────────→ Adam
                                                                │
                                                    + 解耦权重衰减 → AdamW
```

### 11.2 SGD 实现

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                p.data -= lr * p.grad.data
                # θ ← θ - η·∇L
```

**问题**：所有参数用相同的学习率，但不同参数的梯度量级可能差异很大。

### 11.3 AdaGrad 实现

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                state = self.state[p]

                # 累积梯度平方和
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)
                state["g2"] = g2

                # 自适应学习率更新
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
                # θ ← θ - η · g / √(Σg²ᵢ + ε)
```

**关键思想**：
- 梯度大的参数 → g2 累积快 → 学习率自动减小
- 梯度小的参数 → g2 累积慢 → 学习率相对保持较大
- 实现了**每个参数自适应的学习率**

### 11.4 优化器的内存开销

| 优化器 | 每参数额外状态 | 每参数总内存 (float32) |
|--------|---------------|----------------------|
| SGD | 0 | 4+4 = 8 bytes (参数+梯度) |
| SGD+Momentum | 1 (动量) | 4+4+4 = 12 bytes |
| AdaGrad | 1 (g²累积) | 4+4+4 = 12 bytes |
| Adam/AdamW | 2 (一阶矩+二阶矩) | 4+4+4+4 = 16 bytes |

### 11.5 完整内存核算

以 Cruncher 模型为例：

```python
D = 64, num_layers = 2, B = 8

# 参数: (D² × num_layers) + D = 64² × 2 + 64 = 8256
num_parameters = 8256

# 激活值: B × D × num_layers = 8 × 64 × 2 = 1024
num_activations = 1024

# 梯度: 与参数数量相同 = 8256
num_gradients = 8256

# 优化器状态 (AdaGrad): 与参数数量相同 = 8256
num_optimizer_states = 8256

# 总内存 (float32, 4 bytes each)
total_memory = 4 * (8256 + 1024 + 8256 + 8256)  # = 103,168 bytes
```

**推广到 Transformer**：

| 组件 | 内存 (bytes) | 说明 |
|------|-------------|------|
| 参数 | 4N 或 2N (bf16) | N = 参数量 |
| 梯度 | 4N 或 2N (bf16) | 与参数同形 |
| 优化器状态 (Adam) | 8N | 一阶矩 + 二阶矩 |
| 激活值 | 取决于 B, L, d | Batch size × Seq length × Hidden dim |
| **总计** | ~16N + 激活值 | 这就是 "信封背面" 中的 16 bytes/param |

### 11.6 训练一步的 FLOPs

```python
flops_per_step = 6 * B * num_parameters
# 6 = 前向(2) + 反向(4)
```

---

## 12. 训练循环与检查点

### 12.1 完整训练循环

```python
def train(model, get_batch, num_steps, lr):
    optimizer = SGD(model.parameters(), lr=lr)

    for t in range(num_steps):
        # 1. 获取数据
        x, y = get_batch(B=batch_size)

        # 2. 前向传播 (计算 loss)
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)

        # 3. 反向传播 (计算梯度)
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 清零梯度 (重要！)
        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True: 将 grad 设为 None 而非 0
        # 好处: 节省内存 (不需要存全零张量)
```

### 12.2 检查点 (Checkpointing)

**为什么需要？** 训练大模型需要数天到数月，期间硬件故障几乎是必然的。

```python
# 保存检查点
checkpoint = {
    "model": model.state_dict(),         # 模型参数
    "optimizer": optimizer.state_dict(),  # 优化器状态 (动量等)
}
torch.save(checkpoint, "model_checkpoint.pt")

# 加载检查点
loaded = torch.load("model_checkpoint.pt")
model.load_state_dict(loaded["model"])
optimizer.load_state_dict(loaded["optimizer"])
```

**最佳实践**：
- 定期保存 (如每 1000 步)
- 保留最近的 N 个检查点
- 同时保存优化器状态（否则恢复后训练动态会不同）
- 还可以保存：当前步数、学习率调度器状态、随机种子等

---

## 13. 混合精度训练

### 13.1 精度与效率的权衡

| 精度 | 准确性/稳定性 | 内存 | 计算速度 |
|------|-------------|------|----------|
| 高 (float32) | ✅ 好 | ❌ 多 | ❌ 慢 |
| 低 (bf16/fp8) | ⚠️ 有风险 | ✅ 少 | ✅ 快 |

### 13.2 混合精度策略

**核心思想**：不同组件使用不同精度。

```
一个具体方案:
┌──────────────────────┬─────────┐
│ 组件                  │ 精度     │
├──────────────────────┼─────────┤
│ 前向传播 (激活值)     │ bf16/fp8 │  ← 速度快，内存少
│ 参数 (master copy)   │ float32  │  ← 精确累积
│ 梯度                  │ float32  │  ← 避免精度损失
│ 优化器状态            │ float32  │  ← 精确累积
└──────────────────────┴─────────┘
```

### 13.3 PyTorch AMP (Automatic Mixed Precision)

PyTorch 提供了自动混合精度训练库，自动决定哪些操作用低精度、哪些用高精度：

```python
# 使用 torch.cuda.amp
scaler = torch.cuda.amp.GradScaler()

for batch in data_loader:
    with torch.cuda.amp.autocast():  # 自动选择精度
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 13.4 NVIDIA Transformer Engine

更进一步——H100 上可以用 FP8 训练线性层：
- 前向：用 E4M3 (精度优先)
- 反向：用 E5M2 (范围优先)

---

## 14. 核心公式总结

### 内存公式

```
模型内存 ≈ (参数 + 梯度 + 优化器状态) × bytes_per_value + 激活值
         ≈ 16N (AdamW, float32) + 激活值
```

### 计算公式

```
训练一步 FLOPs = 6 × B × N
其中: B = tokens 数, N = 参数量
      6 = 前向(2) + 反向(4)
```

### 时间公式

```
训练时间 = 总 FLOPs / (GPU数 × 单GPU FLOP/s × MFU)
总 FLOPs = 6 × N × D  (N=参数量, D=总训练 token 数)
```

### MFU 公式

```
MFU = 实际 FLOP/s / 理论峰值 FLOP/s
    目标: ≥ 50%
```

### 矩阵乘法

```
(B×D) @ (D×K) → 2BDK FLOPs
```

---

## 总结

Lecture 02 建立了 LLM 训练的**定量思维框架**：

1. **内存核算**：参数 + 梯度 + 优化器状态 + 激活值 → 决定能训练多大的模型
2. **计算核算**：6ND FLOPs → 决定训练需要多长时间
3. **效率衡量**：MFU → 衡量你用了多少 GPU 的理论能力
4. **精度选择**：float32 / bfloat16 / fp8 的权衡，混合精度是最佳实践

**核心思维方式**：在做任何设计决策之前，先做"信封背面"估算——这是 LLM 工程师区别于普通开发者的关键技能。

下一讲：Transformer 架构详解
