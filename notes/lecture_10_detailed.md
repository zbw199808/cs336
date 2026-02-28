# Lecture 10 深入详细学习笔记：推理优化

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_10.py` (可执行讲义)

---

## 目录

1. [推理全景](#1-推理全景)
2. [算术强度分析](#2-算术强度分析)
3. [推理的算术强度](#3-推理的算术强度)
4. [吞吐量与延迟](#4-吞吐量与延迟)
5. [减少 KV Cache 大小](#5-减少-kv-cache-大小)
6. [替代 Transformer 的架构](#6-替代-transformer-的架构)
7. [量化 (Quantization)](#7-量化-quantization)
8. [模型剪枝 (Pruning)](#8-模型剪枝-pruning)
9. [投机采样 (Speculative Sampling)](#9-投机采样-speculative-sampling)
10. [连续批处理 (Continuous Batching)](#10-连续批处理-continuous-batching)
11. [分页注意力 (PagedAttention)](#11-分页注意力-pagedattention)

---

## 1. 推理全景

### 1.1 推理出现的场景

| 场景 | 描述 |
|------|------|
| 实际使用 | 聊天机器人、代码补全、批量数据处理 |
| 模型评估 | 指令遵循等评测 |
| 测试时计算 | Thinking/推理需要更多推理 |
| RL 训练 | 采样生成 → 打分 → 更新 |

### 1.2 效率至关重要

> 训练是一次性成本，推理重复无数次。

- OpenAI 每天处理超 1000 亿 tokens
- Cursor 已写超 10 亿行代码

### 1.3 关键指标

| 指标 | 定义 | 关注场景 |
|------|------|---------|
| TTFT | Time-to-first-token | 交互式应用 |
| 延迟 | 秒/token | 交互式应用 |
| 吞吐量 | tokens/秒 | 批处理应用 |

### 1.4 训练 vs 推理的关键差异

```
训练: 看到所有 tokens，可并行化序列 (matmul) → 计算密集型
推理: 必须顺序生成，难以充分利用计算 → 内存密集型
```

---

## 2. 算术强度分析

### 2.1 矩阵乘法的算术强度

设 X (B×D) 和 W (D×F) 的矩阵乘法：

```
FLOPs:           2BDF
内存传输 (bytes): 2BD + 2DF + 2BF  (读 X, 读 W, 写 Y)
算术强度:        2BDF / (2BD + 2DF + 2BF)
```

**当 B << D, F 时**，简化为：

$$\text{算术强度} \approx B$$

### 2.2 H100 的加速器强度

```
FLOPs/s:          989 × 10¹² (bf16 Tensor Core)
内存带宽:         3.35 × 10¹² bytes/s
加速器强度:       989/3.35 ≈ 295
```

**判断规则**：
- 算术强度 > 295 → **计算瓶颈** (好！GPU 充分利用)
- 算术强度 < 295 → **内存瓶颈** (差！GPU 在等数据)

**结论**：计算受限当且仅当 B > 295

### 2.3 极端情况

**B = 1（矩阵-向量乘）**：
- 算术强度 = 1
- 严重内存瓶颈
- **这正是推理中逐 token 生成的情况！**

---

## 3. 推理的算术强度

### 3.1 推理的两个阶段

| 阶段 | 描述 | 特点 |
|------|------|------|
| **Prefill** | 编码 prompt | 可并行（如训练） |
| **Generation** | 逐 token 生成 | 必须顺序 |

### 3.2 KV Cache

**朴素推理**：生成每个 token 时重新处理整个历史 → O(T³) FLOPs

**KV Cache 优化**：存储之前计算的 Key 和 Value → O(T²) FLOPs

```
KV Cache 内容:
  对每个序列(B)、token(S)、层(L)、头(K) 存储 H 维向量
  大小: B × S × K × H × L × 2 (K和V) × 2 (bf16)
```

### 3.3 MLP 层的算术强度

```
FLOPs:              6BTDF
传输 bytes:         4BTD + 4BTF + 6DF
```

简化（假设 BT << D, F）：**算术强度 ≈ BT**

| 阶段 | T | 算术强度 | 状态 |
|------|---|---------|------|
| Prefill | S (大) | BS | 可计算受限 |
| Generation | 1 | B | 需 B > 295 才计算受限 |

### 3.4 Attention 层的算术强度

```
FLOPs:              4BSTD
传输 bytes:         4BSD + 4BTD
算术强度:           ST / (S + T)
```

| 阶段 | T | 算术强度 | 状态 |
|------|---|---------|------|
| Prefill | S | S/2 | 可计算受限 |
| Generation | 1 | S/(S+1) < 1 | **永远内存受限！** |

### 3.5 关键洞察：为什么 Attention 不受益于 batching？

```
MLP 层:       每个序列共享 Wup, Wgate, Wdown → batching 分摊权重读取
Attention 层:  每个序列有独立的 KV cache → 增大 B 也增大传输量
```

**总结**：
- Prefill = 计算瓶颈 (好)
- Generation = 内存瓶颈 (差)
- MLP 的强度 = B（batching 有帮助）
- Attention 的强度 < 1（batching 无帮助）

---

## 4. 吞吐量与延迟

### 4.1 理论计算（Llama 2 13B on H100）

**假设**：完美重叠计算与通信，忽略开销。

```
内存 = B × KV_cache_per_seq + 参数大小
延迟 = 内存 / 内存带宽
吞吐量 = B / 延迟
```

### 4.2 不同 batch size 的权衡

| B | 内存 | 延迟 | 吞吐量 |
|---|------|------|--------|
| 1 | ~26 GB | 低 | 低 |
| 64 | ~43 GB | 中 | 中 |
| 256 | ~96 GB | 高 | 高但收益递减 |

**B=256 时已经超出单 GPU 内存，且吞吐量增益递减。**

### 4.3 权衡关系

1. 小 batch → 好延迟、差吞吐量
2. 大 batch → 好吞吐量、差延迟

**简单并行**：启动 M 个模型副本 → 延迟不变、吞吐量 ×M

**复杂并行**：分片模型和 KV cache → 参见推理并行文献

### 4.4 TTFT 优化

- Prefill 阶段使用**小 batch** → 更快 TTFT
- Generation 阶段使用**大 batch** → 更高吞吐量

---

## 5. 减少 KV Cache 大小

### 5.1 GQA (Grouped-Query Attention)

```
MHA: K = N (每个 query head 有对应的 KV head)
MQA: K = 1 (所有 query heads 共享一个 KV head)
GQA: 1 < K < N (折中)
```

**效果**：KV cache 缩小 N/K 倍

**Llama 2 13B 示例**：

| 配置 | K | B | 内存 | 延迟 | 吞吐量 |
|------|---|---|------|------|--------|
| 原始 MHA | 40 | 64 | 大 | - | 基准 |
| GQA | 8 | 64 | 减少 | 减少 | 提高 |
| GQA + 大 batch | 8 | 256 | 可行 | 稍增 | 大幅提高 |

**准确率**：GQA 只有轻微下降。

### 5.2 MLA (Multi-head Latent Attention)

**核心思想**：将 KV 投影到低维潜在空间。

```
标准: 缓存 K (N×H 维) + V (N×H 维)
MLA:  缓存 c_KV (C 维), 然后 K = W_UK · c_KV, V = W_UV · c_KV

DeepSeek V2: N×H = 16384 → C = 512 (32× 压缩)
```

**RoPE 兼容性问题**：
- 加额外 64 维用于旋转位置编码
- 总缓存: 512 + 64 = 576 维

**准确率**：MLA 略优于 MHA（同时更便宜！）

### 5.3 CLA (Cross-Layer Attention)

**思想**：跨层共享 KV（类似 GQA 跨 head 共享）

效果：改善准确率-KV cache 大小的 Pareto 前沿。

### 5.4 Local Attention

**思想**：只关注局部上下文

```
优点: KV cache 不随序列长度增长！
缺点: 可能损害准确率

解决方案: 交替使用 local 和 global attention
Example: Character.AI 每 6 层使用 1 层 global attention
```

### 5.5 总结

减少 KV cache 的方法：
1. **低维 KV 表示**: GQA, MLA, 共享 KV cache
2. **Local attention**: 部分层使用

---

## 6. 替代 Transformer 的架构

### 6.1 状态空间模型 (SSM)

| 模型 | 描述 |
|------|------|
| **S4** | 基于经典状态空间模型，擅长合成长序列任务 |
| **Mamba** | SSM 参数依赖于输入，1B 规模匹配 Transformer |
| **Jamba** | 交错 Transformer-Mamba 层 (1:7), 52B MoE |
| **BASED** | 线性注意力 + 局部注意力 |
| **MiniMax-01** | 线性注意力 + 全注意力, 456B MoE |

**核心优势**：用 O(1) 状态替代 O(T) KV cache → 推理效率大幅提升。

**但**：仍需要一些全注意力层来处理关联召回等任务。

### 6.2 扩散模型

- 并行生成所有 token（非自回归）
- 从随机噪声出发，迭代精炼
- Inception Labs 的结果：编码基准上比自回归更快

---

## 7. 量化 (Quantization)

### 7.1 精度谱

| 格式 | 字节数 | 范围 | 用途 |
|------|--------|------|------|
| fp32 | 4 | ±3.4×10³⁸ | 训练参数/优化器 |
| bf16 | 2 | ±3.4×10³⁸ | 推理默认 |
| fp8 (e4m3) | 1 | [-240, 240] | H100 训练可用 |
| int8 | 1 | [-128, 127] | 推理专用 |
| int4 | 0.5 | [-8, 7] | 激进量化 |

### 7.2 量化方法

- **QAT (Quantization-Aware Training)**：训练时加入量化，但不易扩展
- **PTQ (Post-Training Quantization)**：在样本数据上确定缩放因子和零点

### 7.3 LLM.int8()

**问题**：大网络中出现**异常值 (outliers)**，破坏标准量化。

**解决方案**：混合精度——异常值用 fp16 处理，其余用 int8。

效果：在 BLOOM 等模型上基本无损，但比 fp16 慢 15-23%。

### 7.4 AWQ (Activation-aware Weight Quantization)

**思想**：根据激活值选择 0.1-1% 的权重保持高精度。

```
fp16 → int3: 4× 内存减少, 3.2× 加速
```

---

## 8. 模型剪枝 (Pruning)

### 8.1 NVIDIA 的方法

```
1. 在校准数据集 (1024 样本) 上识别重要的 {层, 头, 隐藏维度}
2. 移除不重要的组件得到更小模型
3. 从原始模型蒸馏到剪枝模型
```

---

## 9. 投机采样 (Speculative Sampling)

### 9.1 核心洞察

```
检验 (Prefill): 并行处理 tokens → 快 (计算瓶颈)
生成:           逐 token 生成   → 慢 (内存瓶颈)

→ 检验比生成快得多！
```

### 9.2 算法

```
1. 使用便宜的草稿模型 p 猜测 K 个 token (如 K=4)
2. 用目标模型 q 并行验证（利用 prefill 的速度）
3. 接受 or 拒绝：修改后的拒绝采样
```

### 9.3 数学保证

**关键性质：保证是目标模型的精确采样！**

**二词表证明**（vocabulary = {A, B}）：

```
假设 p(A) > q(A), 因此 p(B) < q(B)

P[采样 A] = p(A) × (q(A)/p(A)) + p(B) × 1 × 0 = q(A) ✓
P[采样 B] = p(B) × 1 + p(A) × (1-q(A)/p(A)) × 1 = q(B) ✓
```

### 9.4 实践配置

| 目标模型 | 草稿模型 |
|---------|---------|
| 70B | 8B |
| 8B | 1B |

**改进方向**：
- **Medusa**：草稿模型并行生成多个 token
- **EAGLE**：草稿模型利用目标模型的高层特征

---

## 10. 连续批处理 (Continuous Batching)

### 10.1 问题

```
训练: 固定的 [batch_size × seq_length] 矩阵
推理: 请求在不同时间到达和完成 → 不规则数组
```

**静态批处理的问题**：必须等一个批次全部完成才能处理新请求。

### 10.2 解决方案：迭代级调度 (Orca)

- 逐步解码
- 新请求到达时**立即加入**当前批次
- 不需要等到整批完成

### 10.3 选择性批处理

不同序列长度的处理：

```
Attention 计算: 各序列分别处理 (不同 KV cache 长度)
非 Attention 计算: 拼接所有序列 → [3+9+5, H] 统一处理
```

---

## 11. 分页注意力 (PagedAttention)

### 11.1 旧方法的问题

```
请求到来 → 预分配 KV cache 空间 (按最大长度)
问题:
  - 内部碎片: 实际生成远少于最大长度
  - 外部碎片: 分配之间的间隙
```

### 11.2 PagedAttention (vLLM)

**灵感**：操作系统的虚拟内存分页机制。

```
将 KV cache 分成非连续的 "块" (blocks)
每个块存储固定数量 token 的 KV
使用页表 (page table) 管理逻辑→物理块映射
```

### 11.3 共享与复用

多种跨序列共享 KV cache 的场景：

| 场景 | 机制 |
|------|------|
| 共享系统 prompt | 共享前缀的 KV cache 块 |
| 多响应采样 | 共享 prompt 的 KV blocks |
| 分叉生成 | 写时复制 (Copy-on-Write) |

### 11.4 其他 vLLM 优化

- 融合 block 读取和注意力的 kernel
- 使用最新的 FlashAttention、FlashDecoding
- CUDA Graphs 避免 kernel 启动开销

---

## 总结

### 推理优化的完整图景

```
理解工作负载:
  - Prefill (计算瓶颈) vs Generation (内存瓶颈)
  - 算术强度分析确定瓶颈

有损优化 (减少复杂度):
  - 架构: GQA, MLA, CLA, Local Attention
  - 新架构: SSM, 扩散模型
  - 量化: fp8, int8, int4 (AWQ)
  - 剪枝 + 蒸馏

无损优化 (保持精确):
  - 投机采样 (精确采样保证)

系统优化 (处理动态工作负载):
  - 连续批处理
  - 分页注意力 (vLLM)
```

### 核心原则

| 原则 | 体现 |
|------|------|
| 推理是内存瓶颈 | 减少 KV cache, 量化减少内存 |
| 检验比生成快 | 投机采样利用这一不对称性 |
| 借鉴系统设计 | 分页 (OS), 推测执行 (CPU) |
| 新架构有巨大潜力 | O(1) 状态替代 O(T) KV cache |

---

下一讲：Lecture 11 — Scaling 详细案例
