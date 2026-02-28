# Lecture 04 深入详细学习笔记：混合专家模型 (Mixture of Experts)

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 4 - MoEs.pdf` (47 页幻灯片)

---

## 目录

1. [MoE 是什么](#1-moe-是什么)
2. [为什么 MoE 越来越流行](#2-为什么-moe-越来越流行)
3. [MoE 的架构设计](#3-moe-的架构设计)
4. [路由函数详解](#4-路由函数详解)
5. [训练目标与负载均衡](#5-训练目标与负载均衡)
6. [训练稳定性与微调挑战](#6-训练稳定性与微调挑战)
7. [Upcycling：从 Dense 到 MoE](#7-upcycling从-dense-到-moe)
8. [DeepSeek MoE V1→V2→V3 演进](#8-deepseek-moe-v1v2v3-演进)
9. [MLA 与多步预测 (MTP)](#9-mla-与多步预测-mtp)

---

## 1. MoE 是什么

### 1.1 核心思想

**用多个小型前馈网络 (experts) + 选择层 (router) 替代一个大型前馈网络。**

```
标准 Transformer FFN:
x → [大型 FFN] → output

MoE FFN:
x → [Router] → 选择 Top-K 个 Expert
         ├→ [Expert 1] ──┐
         ├→ [Expert 2] ──┼→ 加权求和 → output
         └→ [Expert K] ──┘
```

**关键属性**：
- 可以增加专家数量而**不增加 FLOPs**
- 每个 token 只激活一小部分专家
- 总参数量远大于活跃参数量

### 1.2 参数量 vs 活跃参数量

| | Dense 模型 | MoE 模型 |
|---|-----------|----------|
| 总参数量 | N | N × E (E = 专家数) |
| 每 token 活跃参数量 | N | N × K/E (K = Top-K) |
| FLOPs / token | ~2N | ~2N × K/E |

---

## 2. 为什么 MoE 越来越流行

### 2.1 四大优势

1. **相同 FLOPs，更多参数 → 更好性能** (Fedus et al 2022)
2. **训练更快**：相比同性能的 dense 模型 (OLMoE 实验数据)
3. **与 dense 模型高度竞争**：最强的开源模型很多是 MoE
4. **天然可并行**：每个 FFN 专家可以放在不同设备上

### 2.2 实际成果

**西方模型**：
- Mixtral, DBRX, Grok — 目前最高性能的开源模型很多是 MoE

**中国模型**：
- Qwen MoE 系列 — 小规模上做了大量 MoE 工作
- DeepSeek 系列 — 提供了优秀的 MoE 消融实验
- DeepSeek V3 (671B 总参数, 37B 活跃) — MoE 的标杆

### 2.3 为什么 MoE 之前不那么流行？

1. **基础设施复杂**：需要专门的通信和调度机制
2. **多节点优势更明显**：单节点上优势不显著
3. **训练目标启发式**：离散路由不可微分，需要技巧
4. **训练不稳定**：路由器容易引起 loss 尖峰

---

## 3. MoE 的架构设计

### 3.1 替换位置

**最常见**：用 MoE 层替换 FFN/MLP

**较少见**：MoE 用于注意力头 (ModuleFormer, JetMoE)

### 3.2 三大设计维度

| 维度 | 变化空间 |
|------|---------|
| 路由函数 | Top-K, Hash, RL, 线性分配 |
| 专家大小 | 标准, 细粒度 (Fine-grained) |
| 训练目标 | 辅助 loss, 随机扰动, 在线学习 |

### 3.3 主要 MoE 模型的专家配置

| 模型 | 路由专家数 | 活跃数 | 共享专家数 | 细粒度比率 |
|------|-----------|--------|-----------|-----------|
| GShard | 2048 | 2 | 0 | - |
| Switch Transformer | 64 | 1 | 0 | - |
| Mixtral | 8 | 2 | 0 | - |
| DBRX | 16 | 4 | 0 | - |
| Grok | 8 | 2 | 0 | - |
| DeepSeek V1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 MoE | 60 | 4 | 4 | 1/8 |
| **DeepSeek V3** | **256** | **8** | **1** | **1/14** |
| OLMoE | 64 | 8 | 0 | 1/8 |
| LLaMA 4 (Maverick) | 128 | 1 | 1 | 1/2 |

**趋势**：更多、更小的专家 + 少量共享专家

---

## 4. 路由函数详解

### 4.1 路由方案总览

| 方案 | 描述 | 使用情况 |
|------|------|---------|
| **Token-choice Top-K** | Token 选择得分最高的 K 个专家 | **主流** |
| Hash 路由 | 根据 token 的 hash 值分配专家 | 常见基线 |
| Expert-choice | 专家选择自己处理哪些 token | 较少使用 |
| RL 路由 | 用 REINFORCE 学习路由策略 | 早期工作，现不常见 |
| 线性分配 | 将路由建模为匹配问题 | 学术研究 |

### 4.2 Top-K 路由的具体实现

```python
# 基本流程
logits = x @ W_router           # (batch × seq, num_experts)
scores = softmax(logits)         # 概率分布
top_k_indices = topk(scores, k)  # 选择 Top-K
weights = softmax(scores[top_k_indices])  # 归一化权重

output = sum(weights[i] * Expert_i(x) for i in top_k_indices)
```

**两种 softmax 位置的差异**：

| 方案 | 描述 | 使用模型 |
|------|------|---------|
| Softmax → TopK | 先 softmax，再取 TopK | DeepSeek V1/V2, Grok, Qwen |
| TopK → Softmax | 先取 TopK，再对选中的做 softmax | **Mixtral, DBRX, DeepSeek V3** |

### 4.3 DeepSeek 的创新：细粒度 + 共享专家

**细粒度专家 (Fine-grained)**：把一个大专家拆成多个小专家
- 更灵活的组合能力
- 更均匀的负载

**共享专家 (Shared)**：不参与路由，所有 token 都经过
- 捕获通用知识
- 稳定训练

```
DeepSeek V3 的设计:
每个 token → 1 个共享专家 (always on)
           + Top-8 (从 256 个路由专家中选)
```

### 4.4 消融实验结论

- **更多细粒度专家**：一致性改善 (DeepSeek 和 OLMoE 都确认)
- **共享专家**：DeepSeek 确认有帮助，OLMoE 发现无增益 (存在争议)

---

## 5. 训练目标与负载均衡

### 5.1 核心难题

稀疏路由决策是**离散**的，不可微分。如何学习好的路由？

三种方案：

| 方案 | 描述 | 使用情况 |
|------|------|---------|
| RL (REINFORCE) | "正确"的方案 | 梯度方差大，复杂，不常用 |
| 随机扰动 | 给路由 logits 加噪声 | Shazeer 2017, Fedus 2022 |
| **启发式平衡 loss** | 添加辅助 loss 鼓励均匀使用 | **实际主流** |

### 5.2 辅助平衡 Loss (Auxiliary Load Balancing Loss)

**动机**：系统效率要求专家被均匀使用。如果某些专家总是被选中，其他专家的 GPU 就浪费了。

**Switch Transformer 方案**：

```
L_balance = α × N × Σ_i (f_i × p_i)

其中:
- f_i = 分配给专家 i 的 token 比例
- p_i = 路由器给专家 i 的平均概率
- N = 专家数量
- α = 平衡系数
```

**梯度的作用**：对 p_i(x) 的导数与 "专家 i 被使用的频率" 成正比——使用越多，梯度推动使用越少。

### 5.3 DeepSeek V1/V2 的两级平衡

- **Per-Expert Balancing**：同 Switch Transformer
- **Per-Device Balancing**：在设备级别也做均衡（考虑通信效率）

### 5.4 DeepSeek V3 的创新：Auxiliary-Loss-Free

**思路**：不用辅助 loss，而是为每个专家设置一个**偏置项 (bias)**，通过在线学习调整：

```
score_i = router(x)_i + bias_i
如果专家 i 使用过多 → 降低 bias_i
如果专家 i 使用不足 → 增加 bias_i
```

论文称之为 "auxiliary loss free balancing"（虽然实际上不是完全无辅助 loss）。

**效果**：移除辅助平衡 loss 后，token 分布仍然保持合理。

---

## 6. 训练稳定性与微调挑战

### 6.1 训练稳定性

**MoE 路由器容易导致训练不稳定**——路由器的 logits 可能增长过大。

**解决方案**：
- 路由器使用 **Float32** 精度（即使其他部分用 BF16）
- 添加 **Z-Loss** 到路由器（与 Lecture 03 中的 Z-Loss 类似）

### 6.2 微调挑战

稀疏 MoE 在小数据集上容易**过拟合**——因为大量参数、少量微调数据。

**解决方案**：
- Zoph et al：只微调非 MoE 的 MLP 层
- DeepSeek：用大量数据微调 (1.4M SFT 样本)

### 6.3 MoE 的随机性问题

有趣的副作用：MoE 模型可能表现出额外的随机性。

原因：Token dropping (路由时的丢弃) 发生在 batch 级别——**其他请求中的 token 可能导致你的 token 被丢弃！** 这被推测是 GPT-4 随机性的来源之一。

---

## 7. Upcycling：从 Dense 到 MoE

### 7.1 概念

用预训练的 Dense 模型来初始化 MoE 模型，然后继续训练。

```
Dense 模型 (预训练好的)
    ↓ 复制 FFN 到多个专家
MoE 模型 (初始化)
    ↓ 继续训练
MoE 模型 (最终)
```

### 7.2 成功案例

**MiniCPM MoE**：
- 基于 MiniCPM 模型
- Top-K=2, 8 个专家, ~4B 活跃参数
- 仅用 ~520B tokens 就超越基础模型

**Qwen MoE**：
- 从 Qwen 1.8B 初始化
- Top-K=4, 60 个专家 + 4 个共享专家
- 最早确认的成功 upcycling 案例之一

---

## 8. DeepSeek MoE V1→V2→V3 演进

### V1 (16B 总参, 2.8B 活跃)

- 标准 Top-K 路由
- 2 个共享专家 + 64 个路由专家 (细粒度 1/4)
- 标准辅助 loss 均衡 (Expert + Device 级别)

### V2 (236B 总参, 21B 活跃)

新增：
- 160 个路由专家 (细粒度 1/10) + 2 个共享专家，6 个活跃
- **通信均衡 loss**：均衡进出设备的通信量
- **Top-M 设备路由**：先选设备，再在设备内选专家

### V3 (671B 总参, 37B 活跃)

新增：
- 258 个路由专家 + 1 个共享专家，8 个活跃
- **Sigmoid + Softmax TopK + TopM**：更精细的路由
- **Auxiliary-loss-free balancing**：用在线学习的 per-expert bias 替代辅助 loss
- **Sequence-wise auxiliary loss**：序列级辅助 loss

---

## 9. MLA 与多步预测 (MTP)

### 9.1 Multi-head Latent Attention (MLA)

**核心思想**：将 Q, K, V 表示为低维"潜在"激活的函数。

```
标准 MHA:
KV Cache 存储: K 和 V (每头 d_head 维)

MLA:
x → 低维压缩 → c_KV (潜在表示)
c_KV → W_UK → K
c_KV → W_UV → V

KV Cache 只需存储 c_KV！远小于完整的 K, V
```

**好处**：KV Cache 大幅缩小，推理效率提升

**难点**：RoPE 与 MLA 的兼容性

```
Without RoPE:  ⟨Q, K⟩ = ⟨h·W_Q, W_UK·c_KV⟩ = ⟨h·(W_Q·W_UK), c_KV⟩
→ 可以将 W_UK 合并到 Q 投影中，只缓存 c_KV

With RoPE:  ⟨Q·R_q, R_k·W_UK·c_KV⟩ → R_k 阻止了合并！
```

**解决方案**：保留少量非潜在的 key 维度专门用于旋转 (RoPE)，其余维度使用 MLA 压缩。

### 9.2 Multi-Token Prediction (MTP)

在训练时不仅预测下一个 token，还用轻量级辅助模型预测多步：

```
主模型: x_1, x_2, ..., x_t → 预测 x_{t+1}
辅助头1: → 预测 x_{t+2}
辅助头2: → 预测 x_{t+3}
```

DeepSeek V3 只做了 1 步 MTP。参见论文消融实验。

---

## 总结

| 要点 | 内容 |
|------|------|
| MoE 核心 | 利用稀疏性——不是所有输入都需要全部模型 |
| 路由 | 离散路由很难，但 Top-K 启发式有效 |
| 均衡 | 辅助 loss 或 per-expert bias 确保均匀利用 |
| 趋势 | 更多更小的专家、共享专家、auxiliary-loss-free |
| 现状 | 大量经验证据表明 MoE 有效且性价比高 |

---

下一讲：Lecture 05 — GPU 硬件详解
