# Lecture 16 深入详细学习笔记：RLVR (可验证奖励的强化学习)

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `nonexecutable/2025 Lecture 16 - RLVR.pdf` (67 页幻灯片)

---

## 目录

1. [从 RLHF 到 RLVR](#1-从-rlhf-到-rlvr)
2. [GRPO：组相对策略优化](#2-grpo组相对策略优化)
3. [GRPO 的理论分析](#3-grpo-的理论分析)
4. [案例研究：DeepSeek R1](#4-案例研究deepseek-r1)
5. [案例研究：Kimi K1.5](#5-案例研究kimi-k15)
6. [案例研究：Qwen 3](#6-案例研究qwen-3)

---

## 1. 从 RLHF 到 RLVR

### 1.1 RLHF 的局限

- 过优化问题：无法在窄域中干净地扩展
- 奖励模型是学到的近似值

### 1.2 RLVR 的优势

- 在**可精确验证**的领域工作
- 直接优化我们真正想要的

```
RLHF: 预训练 → SFT → RLHF → ~GPT-3.5
RLVR: ... → RLVR → ~o1/R1 (推理能力)
```

---

## 2. GRPO：组相对策略优化

### 2.1 为什么需要新算法？

| | PPO | DPO | GRPO |
|---|-----|-----|------|
| 复杂度 | 高 (价值模型、rollouts) | 低 | 中 |
| 数据格式 | 在线 | 成对 (离线) | 在线 |
| 价值函数 | 需要 | 不需要 | 不需要 |

### 2.2 GRPO 核心思想

```
PPO 的改进:
1. 去掉价值函数 / 优势估计
2. 使用 "组内 z-score" 作为优势

对每个 prompt 生成 G 个回复:
  优势 = (reward - mean) / std  (组内归一化)
```

### 2.3 GRPO 实现

```python
# 极简 GRPO (概念伪代码):
for prompt in batch:
    responses = generate(prompt, num_samples=G)
    rewards = [reward_fn(prompt, r) for r in responses]
    advantages = (rewards - mean(rewards)) / (std(rewards) + 1e-4)

    for response, advantage in zip(responses, advantages):
        log_prob = model.log_prob(response | prompt)
        loss = -advantage * log_prob
        # + KL 惩罚项
        loss.backward()
```

---

## 3. GRPO 的理论分析

### 3.1 基线的有效性

**GRPO 的 std 除法不是有效基线**：

```
有效基线: 可以减去任何 state-dependent 项而保持无偏
GRPO 的 std 归一化: 破坏了无偏性

修正版 [Liu et al. 2025]: REINFORCE with leave-one-out
→ 接近 GRPO 但保持无偏
```

### 3.2 长度偏差

GRPO 的 std 归一化 → 上权过于简单或过难的问题

**长度归一化**：
- 不做归一化：倾向更长回复
- 做归一化：修正但可能有其他问题

---

## 4. 案例研究：DeepSeek R1

### 4.1 为什么 R1 重要？

- 性能超越 OpenAI O1
- 开放的 RL recipe（且相当简单）
- 终结了 MCTS/PRM 必要性的猜测
- SFT 洞察（R1-zero 和 distil-R1）

### 4.2 R1-Zero：纯 RL

```
奖励:
  - 准确性奖励 (正确?)
  - 格式奖励 (使用 thinking tags?)
基础模型: DeepSeek-V3
算法: GRPO (无过程监督)
结果: 略差于 OpenAI O1
```

**有趣现象**：
- 训练中 CoT 逐渐变长
- "Aha moment"（但后续分析表明可能被夸大）
  - 长度可能源于有偏目标
  - 基础模型已经有"aha"能力

### 4.3 R1：完整流程

```
DeepSeek-V3 → 推理 SFT → RL (GRPO) → SFT/RLHF

vs R1-Zero 的关键差异:
  1. SFT 初始化 (长 CoT 数据)
  2. 语言一致性奖励
  3. 非可验证奖励 (第二阶段)
```

**SFT for Reasoning**：
- 仅 1K 数学/科学问题 + Gemini/R1 的长 CoT
- 少量样本就能有效引导推理

**RL 步骤**：
- 基本与 R1-Zero 相同
- 额外的语言一致性损失（RL 自然导致混合语言）

**最终 SFT/RLHF**：
- 推理数据：600K 非可验证任务（V3 做裁判）
- 非推理数据：200K V3 SFT 数据
- RLHF 仍用 GRPO

### 4.4 蒸馏

- R1 生成 800K CoT traces
- 教 Qwen 2.5 做推理

### 4.5 失败尝试

- PRM (PRM800K, DeepSeekMath) — 未成功
- MCTS — 未成功

---

## 5. 案例研究：Kimi K1.5

### 5.1 与 R1 同时发布，也超越 O1

### 5.2 RL 算法

```
参考模型的优化目标:
  基于 DPO 推导的策略梯度
  用平方损失作为代理
  带基线的策略梯度 + 正则化
```

**长度控制**：
```
每个 batch 有长度奖励 λ ∈ [-0.5, 0.5]:
  - 较长序列得到负 λ
  - 正确答案被激励变短
  - 错误答案被激励比中位数更短
  (在训练后期才启用)
```

### 5.3 数据与课程

- 难度标签：从简到难
- 按 (1 - success_rate) 采样，避免重复已解决的问题
- 代码：用真实解法生成新测试用例
- 数学：800K 样本训练 CoT 奖励模型

### 5.4 RL 基础设施

RL 的效率挑战：
- 在线 = rollouts → 慢速推理
- 训练和 rollouts 通常需要不同框架
- 长 CoT 使 batch 非常不均匀

---

## 6. 案例研究：Qwen 3

### 6.1 训练流程

```
SFT → 推理 RL → RLHF → 蒸馏
```

### 6.2 SFT + 推理 RL

- 难度过滤（best-of-n，类似 Kimi）
- 移除模型不需要 CoT 就能答对的
- 移除与验证集过于相似的
- 手动过滤 CoT 质量
- **仅用 3995 个样本做 GRPO！**

### 6.3 思考模式融合

```
创新: 混合 thinking 和 non-thinking 数据
  - 用标签控制
  - 通过特殊字符串实现早停
→ 控制 CoT 长度
```

### 6.4 有趣发现

- 数学/STEM 能力在通用 RLHF 后**略有下降**
- 测试时计算 (test-time scaling) 有效

---

## 总结

| 要点 | 内容 |
|------|------|
| 过优化 | RLHF 的问题；RLVR 在窄域更有效 |
| GRPO | 简单（有缺陷），但使 RLVR 成为可能 |
| R1 | 开源 RL recipe，简单有效 |
| Kimi K1.5 | 长度控制、课程学习 |
| Qwen 3 | 仅 ~4K 样本的 GRPO！ |

---

下一讲：Lecture 17 — 策略梯度深入
