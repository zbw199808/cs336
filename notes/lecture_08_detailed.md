# Lecture 08 深入详细学习笔记：分布式训练

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_08.py` (可执行讲义)

---

## 目录

1. [概述：从单 GPU 到多 GPU](#1-概述从单-gpu-到多-gpu)
2. [集体操作 (Collective Operations)](#2-集体操作-collective-operations)
3. [PyTorch 分布式基础设施](#3-pytorch-分布式基础设施)
4. [通信性能基准测试](#4-通信性能基准测试)
5. [数据并行 (Data Parallelism)](#5-数据并行-data-parallelism)
6. [张量并行 (Tensor Parallelism)](#6-张量并行-tensor-parallelism)
7. [流水线并行 (Pipeline Parallelism)](#7-流水线并行-pipeline-parallelism)

---

## 1. 概述：从单 GPU 到多 GPU

### 1.1 统一主题

Lecture 06-07 讨论的是**单 GPU 内部**的优化（kernel fusion、tiling 减少 HBM 访问）。Lecture 08 将视角扩展到**多 GPU 之间**的协作。

```
单 GPU 内:  ALU ↔ SRAM ↔ HBM        → 减少 HBM 读写 (fusion/tiling)
多 GPU 间:  GPU ↔ NVLink ↔ 其他 GPU  → 减少跨 GPU 通信 (replication/sharding)
```

**核心原则不变**：计算单元（compute）远离数据存储，需要精心编排计算以避免数据传输瓶颈。

### 1.2 广义内存层级

从小/快到大/慢：

| 层级 | 范围 | 介质 |
|------|------|------|
| L1 缓存 / 共享内存 | 单 GPU | SRAM |
| HBM | 单 GPU | DRAM |
| NVLink | 单节点多 GPU | 直连 |
| NVSwitch | 多节点多 GPU | 交换机 |

### 1.3 本讲结构

```
Part 1: 分布式编程的基本构件
  - 集体操作 (概念)
  - torch.distributed (实现)
  - 通信带宽测量

Part 2: 分布式训练策略
  - 数据并行   → 按 batch 维度切分
  - 张量并行   → 按 width 维度切分
  - 流水线并行 → 按 depth 维度切分
```

---

## 2. 集体操作 (Collective Operations)

### 2.1 基本概念

集体操作是分布式编程的**概念原语**，源自 1980 年代的并行计算文献。

**核心术语**：
- **World Size**：设备总数（如 4 个 GPU）
- **Rank**：设备编号（如 0, 1, 2, 3）

### 2.2 六大操作

| 操作 | 描述 | 输入→输出 |
|------|------|-----------|
| **Broadcast** | 一个 rank 的数据复制到所有 rank | 1→N |
| **Scatter** | 一个 rank 的数据分片到各 rank | 1→N (分片) |
| **Gather** | 所有 rank 的数据收集到一个 rank | N→1 |
| **Reduce** | 所有 rank 的数据归约到一个 rank | N→1 (聚合) |
| **All-gather** | Gather + 结果广播到所有 rank | N→N (全复制) |
| **Reduce-scatter** | Reduce 后分片到各 rank | N→N (分片) |

### 2.3 组合关系

```
All-reduce = Reduce-scatter + All-gather
```

这是分布式训练中**最常用**的操作——用于梯度同步。

**记忆术**：
- **Reduce** = 执行某种结合律/交换律操作（sum, min, max）
- **Broadcast/Scatter** = Gather 的逆操作
- **All** = 目标是所有设备

### 2.4 PyTorch 代码示例

```python
# All-reduce: 每个 rank 的 tensor 求和后分发给所有 rank
tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank
dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
# 结果: 每个 rank 都拿到相同的求和结果

# Reduce-scatter: 求和后每个 rank 只拿到一部分
input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
output = torch.empty(1, device=get_device(rank))
dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM)
# 结果: rank i 拿到第 i 个元素的总和

# All-gather: 收集所有 rank 的 output
dist.all_gather_into_tensor(output_tensor=full_output, input_tensor=output)
# 验证: all-reduce == reduce-scatter + all-gather
```

---

## 3. PyTorch 分布式基础设施

### 3.1 硬件连接方式

**传统（家用/旧集群）**：
- 同节点 GPU：PCIe 总线（v7.0, 16 lanes → 242 GB/s）
- 跨节点：以太网（~200 MB/s）

**现代数据中心**：
- 同节点：**NVLink** 直连 GPU，绕过 CPU
- 跨节点：**NVSwitch** 直连 GPU，绕过以太网

**H100 NVLink 规格**：
- 每个 H100 有 18 条 NVLink 4.0 链路
- **总带宽 900 GB/s**（双向）
- 对比 HBM 带宽 3.9 TB/s

### 3.2 NCCL (NVIDIA Collective Communication Library)

NCCL 将高层集体操作翻译为底层 GPU 间数据包：

```
用户代码: dist.all_reduce(tensor, op=SUM)
    ↓
NCCL:
  1. 检测硬件拓扑 (节点数、交换机、NVLink/PCIe)
  2. 优化 GPU 间的传输路径
  3. 启动 CUDA kernel 发送/接收数据
```

### 3.3 PyTorch Distributed

```python
# 初始化进程组
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "15623"

if torch.cuda.is_available():
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
else:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

**后端选择**：
- `nccl`：GPU（通过 NCCL）
- `gloo`：CPU

**多进程启动**：使用 `spawn` 函数为每个 rank 启动一个进程。

---

## 4. 通信性能基准测试

### 4.1 All-reduce 带宽测量

```python
def all_reduce(rank, world_size, num_elements):
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # 测量
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    duration = time.time() - start_time

    # 有效带宽计算
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x: 发送+接收
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
```

### 4.2 关键细节

- **Warmup**：首次通信可能因初始化更慢
- **torch.cuda.synchronize()**：等待 GPU 操作完成
- **dist.barrier()**：等待所有进程到达同一点
- **带宽计算**：All-reduce 的总传输量 = 数据大小 × 2 × (world_size - 1)

### 4.3 Reduce-scatter 带宽

```python
# Reduce-scatter 的传输量不需要 2x
sent_bytes = data_bytes * (world_size - 1)
```

---

## 5. 数据并行 (Data Parallelism)

### 5.1 核心思想

**每个 rank 持有完整模型参数，但只处理 batch 的一部分。**

```
Rank 0: data[0:32]   + 完整参数 → 计算梯度 → all-reduce 梯度
Rank 1: data[32:64]  + 完整参数 → 计算梯度 → all-reduce 梯度
Rank 2: data[64:96]  + 完整参数 → 计算梯度 → all-reduce 梯度
Rank 3: data[96:128] + 完整参数 → 计算梯度 → all-reduce 梯度
```

### 5.2 实现代码

```python
def data_parallelism_main(rank, world_size, data, num_layers, num_steps):
    setup(rank, world_size)

    # 1. 切分数据
    local_batch_size = batch_size // world_size
    start = rank * local_batch_size
    data = data[start:start+local_batch_size].to(get_device(rank))

    # 2. 每个 rank 创建完整的模型参数
    params = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # 前向传播 (在本地数据上)
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()

        # 反向传播
        loss.backward()

        # ★ 唯一与标准训练不同的地方：同步梯度 ★
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)

        # 更新参数
        optimizer.step()
```

### 5.3 关键性质

| 属性 | 行为 |
|------|------|
| Loss | 各 rank **不同**（计算在不同数据上） |
| 梯度 | all-reduce 后**相同** |
| 参数 | 始终**相同**（因为梯度相同、初始化相同） |
| 通信量 | 每步 = 2 × 参数量 × (world_size-1) / world_size |

### 5.4 优缺点

**优点**：
- 实现最简单
- 线性加速 batch 处理

**缺点**：
- 每个 rank 存储完整参数 + 优化器状态（内存瓶颈）
- 通信量与参数量成正比

---

## 6. 张量并行 (Tensor Parallelism)

### 6.1 核心思想

**每个 rank 持有每层参数的一部分，处理全部数据。**

```
Rank 0: params[:, 0:256]    → 计算部分激活 → all-gather 拼接完整激活
Rank 1: params[:, 256:512]  → 计算部分激活 → all-gather 拼接完整激活
Rank 2: params[:, 512:768]  → 计算部分激活 → all-gather 拼接完整激活
Rank 3: params[:, 768:1024] → 计算部分激活 → all-gather 拼接完整激活
```

### 6.2 实现代码

```python
def tensor_parallelism_main(rank, world_size, data, num_layers):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    local_num_dim = num_dim // world_size  # 每个 rank 的宽度

    # 每个 rank 只创建 1/world_size 的参数
    params = [get_init_params(num_dim, local_num_dim, rank) for _ in range(num_layers)]

    x = data
    for i in range(num_layers):
        # 计算部分激活: [batch, num_dim] @ [num_dim, local_num_dim]
        x = x @ params[i]   # 结果: [batch, local_num_dim]
        x = F.gelu(x)

        # All-gather: 收集所有 rank 的激活
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank))
                       for _ in range(world_size)]
        dist.all_gather(tensor_list=activations, tensor=x)

        # 拼接成完整激活: [batch, num_dim]
        x = torch.cat(activations, dim=1)
```

### 6.3 通信模式

```
每层需要一次 all-gather:
  输入:  [batch, local_dim] (每个 rank)
  输出:  [batch, num_dim]   (每个 rank)
  通信量: batch × local_dim × (world_size - 1) × element_size
```

### 6.4 对比数据并行

| 维度 | 数据并行 | 张量并行 |
|------|---------|---------|
| 切分维度 | batch | width (隐藏维度) |
| 参数存储 | 全量 (冗余) | 1/N |
| 通信内容 | 梯度 (每步一次) | 激活 (每层一次) |
| 通信时机 | 反向传播后 | 每层前向后 |
| 适用场景 | 模型可放入单 GPU | 单层参数过大 |

---

## 7. 流水线并行 (Pipeline Parallelism)

### 7.1 核心思想

**每个 rank 负责一部分层，数据按流水线方式通过。**

```
Rank 0: Layer 0, Layer 1  →  发送中间激活给 Rank 1
Rank 1: Layer 2, Layer 3  →  接收激活, 继续计算
```

### 7.2 微批次 (Micro-batches)

为了减少**流水线气泡**（Pipeline Bubble），将 batch 切分为多个微批次：

```
时间 →
Rank 0: [μ1] [μ2] [μ3] [μ4]
Rank 1:      [μ1] [μ2] [μ3] [μ4]

没有微批次时:
Rank 0: [全部 batch]
Rank 1:              [全部 batch]  ← Rank 1 大部分时间空闲！
```

### 7.3 实现代码

```python
def pipeline_parallelism_main(rank, world_size, data, num_layers, num_micro_batches):
    setup(rank, world_size)

    local_num_layers = num_layers // world_size
    local_params = [get_init_params(num_dim, num_dim, rank) for _ in range(local_num_layers)]

    # 切分微批次
    micro_batch_size = batch_size // num_micro_batches
    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank))
                         for _ in range(num_micro_batches)]

    for x in micro_batches:
        # 从前一个 rank 接收激活
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 计算本 rank 负责的层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 发送激活给下一个 rank
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)
```

### 7.4 点对点通信

流水线并行使用**点对点**通信（`dist.send` / `dist.recv`），而非集体操作：

| 通信方式 | 操作 | 使用场景 |
|---------|------|---------|
| 集体操作 | all-reduce, all-gather | 数据并行、张量并行 |
| 点对点 | send, recv | 流水线并行 |

### 7.5 未处理的问题

- 通信与计算的重叠（消除气泡）
- 反向传播的实现（homework exercise）
- 更复杂的调度策略（1F1B、Interleaved Pipeline 等）

---

## 总结

### 三种并行策略对比

| | 数据并行 | 张量并行 | 流水线并行 |
|---|---------|---------|-----------|
| 切分维度 | Batch | Width | Depth |
| 每 rank 存储 | 全部参数 | 部分参数 | 部分层 |
| 通信操作 | All-reduce (梯度) | All-gather (激活) | Send/Recv (激活) |
| 通信频率 | 每步一次 | 每层一次 | 每微批次每阶段 |
| 实现复杂度 | 低 | 中 | 高 |
| 扩展性瓶颈 | 内存 (参数冗余) | 通信带宽 | Pipeline bubble |

### 核心要点

- **重计算 vs 存储 vs 通信**的三方权衡始终存在
- 硬件变快但模型也变大 → 层级结构和通信优化永远重要
- 实际系统需要组合多种策略（3D 并行 = DP + TP + PP）
- Jax/TPU 可以自动处理分片（声明式），PyTorch 需要手动构建（命令式）

### 遗留话题

- 更通用的模型（带 Attention）
- 通信与计算的重叠
- 序列并行 (Sequence Parallelism)
- 专家并行 (Expert Parallelism, 用于 MoE)
- ZeRO 优化器状态分片

---

下一讲：Lecture 09 — Scaling Laws 基础
