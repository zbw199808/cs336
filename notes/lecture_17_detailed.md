# Lecture 17 深入详细学习笔记：策略梯度深入 (Policy Gradient Deep Dive)

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_17.py` (可执行讲义)

---

## 目录

1. [语言模型的 RL 设定](#1-语言模型的-rl-设定)
2. [策略梯度推导](#2-策略梯度推导)
3. [基线与方差缩减](#3-基线与方差缩减)
4. [优势函数](#4-优势函数)
5. [GRPO 训练实战](#5-grpo-训练实战)
6. [任务设计：排序问题](#6-任务设计排序问题)
7. [模型设计](#7-模型设计)
8. [奖励计算与 Delta 模式](#8-奖励计算与-delta-模式)
9. [损失函数变体](#9-损失函数变体)
10. [KL 惩罚](#10-kl-惩罚)
11. [实验与结果分析](#11-实验与结果分析)

---

## 1. 语言模型的 RL 设定

### 1.1 MDP 元素映射

将标准 RL 的 MDP 元素映射到语言模型：

| MDP 元素 | 语言模型中的含义 | 特殊性 |
|----------|----------------|--------|
| **状态 s** | prompt + 已生成的回复 | 确定性增长 |
| **动作 a** | 生成下一个 token | 离散动作空间 |
| **奖励 R** | 回复质量评分 | 结果奖励 (outcome reward) |
| **转移 T(s'∣s,a)** | s' = s + a（拼接） | **确定性**转移 |
| **策略 π(a∣s)** | 语言模型本身 | 微调后的 LM |

### 1.2 与机器人学的关键差异

```
机器人学:
  - 转移概率是随机的 (物理世界不确定性)
  - 状态是真实的物理状态
  - 难以做规划/搜索

语言模型:
  - 转移是确定性的 (s' = s + a)
  - 状态是"虚构的" (纯文本拼接)
  - 可以做规划 / test-time compute
  → 极大的灵活性
```

### 1.3 聚焦条件

本讲聚焦于：
- **结果奖励 (Outcome rewards)**：依赖整个回复，非逐步奖励
- **可验证奖励 (Verifiable rewards)**：计算是确定性的（如数学答案对错）
- 折扣和 bootstrapping 不太适用

### 1.4 Rollout 结构

```
一个 rollout/episode/trajectory:
  s → a₁ → a₂ → ... → aₙ → R

目标: 最大化 E[R]
  期望取自 prompt 分布 p(s) 和策略 π(a|s)
```

---

## 2. 策略梯度推导

### 2.1 目标函数

为简化符号，令 **a** 表示**整个回复**（所有 token 的序列）。

$$E[R] = \int p(s) \pi(a|s) R(s,a) \, ds \, da$$

### 2.2 梯度推导（Log-trick）

```
∇ E[R] = ∫ p(s) ∇π(a|s) R(s,a)           # 直接求导
        = ∫ p(s) π(a|s) ∇log π(a|s) R(s,a)  # log-trick: ∇π = π·∇log π
        = E[∇log π(a|s) R(s,a)]              # 变回期望形式
```

**Log-trick 推导**：

$$\nabla \pi(a|s) = \pi(a|s) \cdot \nabla \log \pi(a|s)$$

因为 $\nabla \log f = \frac{\nabla f}{f}$，所以 $\nabla f = f \cdot \nabla \log f$。

### 2.3 朴素策略梯度

```python
# 朴素策略梯度算法：
# 1. 采样 prompt s
# 2. 采样回复 a ~ π(a|s)
# 3. 基于 ∇log π(a|s) · R(s,a) 更新参数
```

**直觉理解**：
- 本质上和 SFT 相同，但加权了 R(s,a)
- 数据集随策略变化而动态变化（在线学习）

### 2.4 稀疏奖励的挑战

当 R(s,a) ∈ {0, 1}（对/错）：
- 朴素策略梯度**只在正确回复上更新**（R=0 时梯度为 0）
- 像"在正确样本上做 SFT"
- 如果大部分回复是错的 → 很少有梯度信号 → **方差极高**

对比 RLHF：奖励模型提供连续分数，信号更丰富。

---

## 3. 基线与方差缩减

### 3.1 问题：原始奖励的误导

**经典两状态例子**：

```
s1: a1 → 奖励 11, a2 → 奖励 9
s2: a1 → 奖励 0,  a2 → 奖励 2

问题: s1 → a2 的奖励是 9, s2 → a2 的奖励是 2
      9 > 2, 但在 s1 中 a2 是差的 (因为 a1 得 11)
      在 s2 中 a2 是好的 (因为 a1 得 0)
```

原始奖励无法区分"在该状态下是否是好动作"。

### 3.2 基线的引入

**关键思想**：减去与状态相关的基线 b(s)。

$$\nabla E[R] = E[\nabla \log \pi(a|s) \cdot (R(s,a) - b(s))]$$

**为什么无偏**？因为 $E[\nabla \log \pi(a|s) \cdot b(s)] = 0$：

$$\int \pi(a|s) \nabla \log \pi(a|s) \cdot b(s) \, da = b(s) \int \nabla \pi(a|s) \, da = b(s) \nabla 1 = 0$$

### 3.3 方差缩减效果

```python
# 无基线
naive_variance = std([11, 9, 0, 2])  # ≈ 5.066

# 有基线 b(s1) = 10, b(s2) = 1
baseline_variance = std([11-10, 9-10, 0-1, 2-1])  # ≈ 1.291

# 方差从 5.066 降到 1.291！
```

### 3.4 最优基线

**理论最优**（单参数模型）：

$$b^*(s) = \frac{E[(\nabla \pi(a|s))^2 R | s]}{E[(\nabla \pi(a|s))^2 | s]}$$

这很难计算，实际中常用**启发式基线**：

$$b(s) = E[R|s] \approx \text{同一 prompt 下多个回复的平均奖励}$$

---

## 4. 优势函数

### 4.1 价值函数与 Q 函数

| 函数 | 定义 | 含义 |
|------|------|------|
| V(s) | E[R∣s] | 从状态 s 出发的期望奖励 |
| Q(s,a) | E[R∣s,a] | 从状态 s 采取动作 a 的期望奖励 |

**注意**：在结果奖励设定下，如果 a 是整个回复，那么 Q(s,a) = R(s,a)。

### 4.2 优势函数的定义

$$A(s,a) = Q(s,a) - V(s)$$

**直觉**：动作 a 比"平均水平"好多少。

### 4.3 与基线的关系

当 b(s) = E[R|s] = V(s) 时：

$$R(s,a) - b(s) = Q(s,a) - V(s) = A(s,a)$$

**基线化的奖励就是优势函数！**

### 4.4 δ 的选择

策略梯度的一般形式：

$$\text{估计: } \nabla \log \pi(a|s) \cdot \delta$$

其中 δ 有多种选择：

| δ | 名称 | 偏差 | 方差 |
|---|------|------|------|
| R(s,a) | 原始奖励 | 无偏 | 高 |
| R(s,a) - b(s) | 基线化奖励 | 无偏 | 较低 |
| A(s,a) | 优势 | 无偏 | 低 |
| (R - mean)/std | 归一化奖励 | **有偏** | 最低 |

---

## 5. GRPO 训练实战

### 5.1 GRPO 回顾

Group Relative Policy Optimization 的核心简化：
- 去掉 PPO 中的 critic（价值函数）
- 利用 LM 设定的**组结构**：每个 prompt 生成多个回复
- 组内回复提供了自然的基线 b(s)

### 5.2 完整训练循环

```python
for epoch in range(num_epochs):
    # 1. 如果使用 KL 惩罚，定期冻结参考模型
    if kl_penalty != 0 and epoch % period == 0:
        ref_model = model.clone()

    # 2. 采样回复，计算奖励
    responses = generate_responses(prompts, model, num_responses)
    rewards = compute_reward(prompts, responses, reward_fn)
    deltas = compute_deltas(rewards, mode=deltas_mode)

    # 3. 冻结参考 log_probs（如果使用比率损失）
    if loss_mode != "naive":
        with torch.no_grad():
            old_log_probs = compute_log_probs(prompts, responses, model)

    # 4. 多步内循环优化
    for step in range(num_steps_per_epoch):
        log_probs = compute_log_probs(prompts, responses, model)
        loss = compute_loss(log_probs, deltas, mode=loss_mode,
                           old_log_probs=old_log_probs)
        if kl_penalty != 0:
            loss += kl_penalty * compute_kl_penalty(log_probs, ref_log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. 任务设计：排序问题

### 6.1 任务定义

```
输入 (prompt): n 个数字, 如 [1, 0, 2]
输出 (response): n 个数字, 如 [0, 1, 2] (排好序)
```

简单但足以展示 RL 训练的核心问题。

### 6.2 奖励函数设计

**奖励函数 1：精确匹配距离 (sort_distance_reward)**

```python
def sort_distance_reward(prompt, response):
    ground_truth = sorted(prompt)
    return sum(1 for x, y in zip(response, ground_truth) if x == y)
```

只计算与正确答案完全匹配的位置数。

**奖励函数 2：包含+排序奖励 (sort_inclusion_ordering_reward)**

```python
def sort_inclusion_ordering_reward(prompt, response):
    # 包含奖励：response 中出现了 prompt 中的元素
    inclusion = sum(1 for x in prompt if x in response)
    # 排序奖励：相邻元素是否有序
    ordering = sum(1 for x, y in zip(response, response[1:]) if x <= y)
    return inclusion + ordering
```

### 6.3 奖励函数对比

```
prompt = [3, 1, 0, 2], ground_truth = [0, 1, 2, 3]

response           | distance_reward | inclusion_ordering_reward
[0, 1, 2, 3]       | 4 (全对)        | 4 + 3 = 7 (完美)
[7, 2, 2, 5]       | 0               | 1 + 2 = 3
[0, 3, 1, 2]       | 1               | 3 + 1 = 4  ← 部分信用更多！
```

**关键洞察**：`inclusion_ordering_reward` 给予更多**部分信用**，为 RL 训练提供更丰富的梯度信号。奖励函数的设计对 RL 训练至关重要。

---

## 7. 模型设计

### 7.1 简化模型架构

```python
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, prompt_length, response_length):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 每个位置独立的编码/解码矩阵
        self.encode_weights = Parameter(randn(prompt_length, dim, dim) / sqrt(dim))
        self.decode_weights = Parameter(randn(response_length, dim, dim) / sqrt(dim))
```

### 7.2 前向传播

```
输入 prompts [batch, pos]
  → embedding [batch, pos, dim]
  → encode: einsum(embed, encode_weights) → [batch, dim]  # 折叠所有位置
  → decode: einsum(encoded, decode_weights) → [batch, pos, dim]  # 展开到回复位置
  → logits: einsum(decoded, embedding.weight) → [batch, pos, vocab]
```

**设计特点**：
- 固定长度 prompt 和 response
- **非自回归**：每个位置独立解码
- 输入/输出**共享 embedding**
- 每个位置有独立的参数矩阵 → 捕捉位置信息

### 7.3 回复生成

```python
def generate_responses(prompts, model, num_responses):
    logits = model(prompts)  # [batch, pos, vocab]
    # 对每个 (batch, pos) 独立采样 num_responses 次
    responses = multinomial(softmax(logits), num_samples=num_responses)
    # → [batch, trial, pos]
```

---

## 8. 奖励计算与 Delta 模式

### 8.1 四种 Delta 模式

给定奖励 rewards [batch, trial]，计算 delta 用于梯度更新：

| 模式 | 公式 | 特点 |
|------|------|------|
| `rewards` | δ = R | 原始奖励，正值更新 |
| `centered_rewards` | δ = R - mean(R) | 减去组均值，有正有负 |
| `normalized_rewards` | δ = (R - mean) / std | GRPO 标准做法，归一化 |
| `max_rewards` | δ = R if R==max else 0 | 只保留最优回复 |

### 8.2 实现细节

```python
def compute_deltas(rewards, mode):
    if mode == "rewards":
        return rewards

    if mode == "centered_rewards":
        mean = rewards.mean(dim=-1, keepdim=True)  # 每个 prompt 的均值
        return rewards - mean

    if mode == "normalized_rewards":
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        return (rewards - mean) / (std + 1e-5)

    if mode == "max_rewards":
        max_r = rewards.max(dim=-1, keepdim=True)[0]
        return torch.where(rewards == max_r, rewards, torch.zeros_like(rewards))
```

### 8.3 各模式的直觉

```
假设某 prompt 的 5 个回复奖励为: [3, 5, 5, 2, 4]

rewards:            [3, 5, 5, 2, 4]     全部正值 → 全部加强
centered_rewards:   [-0.8, 1.2, 1.2, -1.8, 0.2]  差的负梯度，好的正梯度
normalized_rewards: [-0.6, 0.9, 0.9, -1.3, 0.15]  缩放后，跨 prompt 可比
max_rewards:        [0, 5, 5, 0, 0]     只加强最优
```

---

## 9. 损失函数变体

### 9.1 三种损失模式

**朴素损失 (naive)**：

$$L = -E[\log \pi(a|s) \cdot \delta]$$

```python
loss = -(log_probs * deltas).mean()
```

**未裁剪比率损失 (unclipped)**：

$$L = -E\left[\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)} \cdot \delta\right]$$

```python
ratios = exp(log_probs - old_log_probs)
loss = -(ratios * deltas).mean()
```

**裁剪比率损失 (clipped)** — PPO/GRPO 核心：

$$L = -E\left[\min\left(\frac{\pi}{\pi_{\text{old}}} \delta, \; \text{clip}\left(\frac{\pi}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) \delta\right)\right]$$

```python
epsilon = 0.01
ratios = exp(log_probs - old_log_probs)
unclipped = ratios * deltas
clipped_ratios = clamp(ratios, 1 - epsilon, 1 + epsilon)
clipped = clipped_ratios * deltas
loss = -min(unclipped, clipped).mean()
```

### 9.2 裁剪的直觉

```
clipped ratio 限制策略更新幅度:

  δ > 0 (好动作):
    ratio 增大 → 好, 但不能超过 1 + ε
    防止过度自信

  δ < 0 (差动作):
    ratio 减小 → 好, 但不能低于 1 - ε
    防止过度惩罚

→ 保持新旧策略的信赖域约束
```

### 9.3 参数冻结的重要性

```python
# 错误做法：p_old 也参与梯度计算
w = tensor(2., requires_grad=True)
p = sigmoid(w)
p_old = sigmoid(w)      # 没有 detach！
ratio = p / p_old       # ratio 的梯度 ≈ 0（分子分母抵消）
ratio.backward()
# w.grad ≈ 0  ← 错误！

# 正确做法：冻结 p_old
w = tensor(2., requires_grad=True)
p = sigmoid(w)
with torch.no_grad():
    p_old = sigmoid(w)  # 作为常数对待
ratio = p / p_old
ratio.backward()
# w.grad 正确反映 p 对 w 的导数
```

**实际代码中的体现**：

```python
# 外循环：冻结旧模型的 log_probs
with torch.no_grad():
    old_log_probs = compute_log_probs(prompts, responses, model)

# 内循环：当前模型的 log_probs 正常计算梯度
log_probs = compute_log_probs(prompts, responses, model)
```

---

## 10. KL 惩罚

### 10.1 动机

- RL 可能使模型"忘记"原始能力
- KL 惩罚保持新策略与参考策略的接近

### 10.2 KL 散度的估计

标准 KL：$KL(p \| q) = E_{x \sim p}[\log(p(x)/q(x))]$

使用**稳健估计器**：

$$KL(p \| q) = E_{x \sim p}\left[\frac{q(x)}{p(x)} - \log\frac{q(x)}{p(x)} - 1\right]$$

这等价于标准 KL，因为 $E_{x \sim p}[q(x)/p(x)] = 1$。

```python
def compute_kl_penalty(log_probs, ref_log_probs):
    # exp(ref - cur) = ref/cur, 即 q/p
    return (exp(ref_log_probs - log_probs)
            - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()
```

### 10.3 实际使用

```python
# 定期冻结参考模型（每 compute_ref_model_period 个 epoch）
if epoch % compute_ref_model_period == 0:
    ref_model = model.clone()

# 加入损失
loss = policy_loss + kl_penalty * kl_term
```

---

## 11. 实验与结果分析

### 11.1 实验设置

```
任务: 排序 3 个数字
Prompts: [[1,0,2], [3,2,4], [1,2,3]]
Vocab size: 5 (0-4)
每个 prompt 采样: 10 个回复
训练: 100 epochs × 10 steps/epoch
优化器: Adam, lr=1e-3
奖励函数: sort_inclusion_ordering_reward
```

### 11.2 实验 1：原始奖励 (rewards + naive)

```
配置: deltas_mode="rewards", loss_mode="naive"
结果: 到最后仍然没有很好地学会排序 (即使在训练集上)
问题: 所有奖励都是正的 → 所有回复都被加强
      好的和差的回复都得到正梯度
```

### 11.3 实验 2：中心化奖励 (centered_rewards + naive)

```
配置: deltas_mode="centered_rewards", loss_mode="naive"
改进:
  1. 次优回复得到负梯度更新 → 被抑制
  2. 如果某 prompt 的所有回复奖励相同 → 不更新 (没信息)
结果: 更好了，但仍然卡在局部最优
```

### 11.4 实验 3：归一化奖励 (normalized_rewards + naive)

```
配置: deltas_mode="normalized_rewards", loss_mode="naive"
结果: 与 centered_rewards 差异不大
原因: 所有回复长度相同，std 归一化影响有限
注: Dr. GRPO 论文建议不做 std 归一化以避免长度偏差
    (在本任务中不是问题，因为回复定长)
```

### 11.5 关键教训

```
1. RL 训练并不简单，容易卡在次优状态
2. 基线的选择至关重要 (rewards → centered → normalized)
3. 超参数需要仔细调整
4. 奖励函数设计影响训练难度
5. 即使在简单任务上，RL 也可能不完美
```

---

## 总结

### 核心要点

| 要点 | 内容 |
|------|------|
| RL 的力量 | 超越人类能力的关键——**如果能度量就能优化** |
| 策略梯度 | 概念清晰，核心挑战是方差缩减 |
| 基线 | 将奖励变为优势函数，大幅降低方差 |
| GRPO | 利用组结构提供自然基线，去掉 critic |
| 裁剪 | 限制策略更新幅度，保持训练稳定 |
| 工程挑战 | RL 系统比预训练复杂得多（推理、多模型管理） |

### 策略梯度家族一览

```
朴素策略梯度
  ↓ + 基线 (方差缩减)
REINFORCE with baseline
  ↓ + 信赖域约束
TRPO
  ↓ + 裁剪代理 (实现简单)
PPO
  ↓ - critic + 组结构基线
GRPO ← 当前 RLVR 的主流选择
```

### 代码结构总结

```
lecture_17.py 的模块化设计:

rl_setup_for_language_models()  → MDP 定义
policy_gradient()               → 理论推导 + 基线 + 优势函数
training_walkthrough()          → GRPO 实战
  ├── simple_task()             → 排序任务 + 奖励函数
  ├── simple_model()            → 模型 + 生成 + delta + loss
  └── experiments()             → 三组对比实验

关键函数:
  sort_distance_reward()             → 精确匹配奖励
  sort_inclusion_ordering_reward()   → 部分信用奖励
  compute_deltas(mode=...)           → 4 种 delta 计算
  compute_loss(mode=...)             → 3 种损失计算
  compute_kl_penalty()               → KL 散度估计
  run_policy_gradient()              → 完整训练循环
```

---

下两讲：
- Junyang Lin (Qwen) 特邀报告
- Mike Lewis (Llama) 特邀报告
