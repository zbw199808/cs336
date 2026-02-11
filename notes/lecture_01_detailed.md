# Lecture 01 深入详细学习笔记：课程概述与分词 (Tokenization)

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_01.py` (可执行讲义)

---

## 目录

1. [为什么需要这门课](#1-为什么需要这门课)
2. [语言模型的工业化](#2-语言模型的工业化)
3. ["更大即不同" 与可迁移的知识](#3-更大即不同与可迁移的知识)
4. [苦涩的教训 (The Bitter Lesson)](#4-苦涩的教训)
5. [语言模型发展全景](#5-语言模型发展全景)
6. [课程五大模块详解](#6-课程五大模块详解)
7. [分词 (Tokenization) 深入剖析](#7-分词-tokenization-深入剖析)
8. [关键论文索引](#8-关键论文索引)

---

## 1. 为什么需要这门课

### 1.1 研究者与底层技术的脱节

课程开篇提出了一个深刻的观察——研究者正在与底层技术渐行渐远：

| 时间线 | 研究者的工作方式 | 抽象层级 |
|--------|------------------|----------|
| **8年前** (~2017) | 自己实现并训练模型 | 最底层 |
| **6年前** (~2019) | 下载预训练模型 (如 BERT) 进行微调 | 中间层 |
| **今天** (2025) | 直接 prompt 商业模型 (GPT-4/Claude/Gemini) | 最高层 |

**为什么这是个问题？**

向上移动抽象层次本身是好事——它提高了生产力。在编程语言、操作系统等领域，抽象层做得很好，你不需要理解晶体管就能写好代码。但语言模型的抽象层存在根本性区别：

1. **抽象是"泄漏的" (Leaky Abstractions)**：与编程语言不同，LLM 的行为无法被精确预测。你不理解底层，就无法解释为什么某个 prompt 有效而另一个无效、为什么模型会产生幻觉、为什么某些任务表现好而某些差。

2. **基础研究仍需拆解整个栈**：如果你想改进 attention 机制、设计新的训练策略、或者探索新的架构，你必须理解从数据处理到 GPU 编程的全栈知识。

### 1.2 核心理念：通过构建来理解

课程的核心哲学是 **"understanding via building"**——不是看论文、不是读教科书，而是从零开始实现每个组件。这种方式确保你不仅知道"是什么"，更知道"为什么"和"怎么做"。

---

## 2. 语言模型的工业化

### 2.1 规模的量级感

课程用一系列令人震撼的数字来说明当今 LLM 的工业化程度：

| 项目 | 规模 |
|------|------|
| GPT-4 参数量 | 据传 **1.8 万亿** (1.8T) |
| GPT-4 训练成本 | 据传 **$100M** |
| xAI Grok 集群 | **200,000 张 H100** GPU |
| Stargate 项目 (OpenAI+NVIDIA+Oracle) | 4年投资 **$500B** |

### 2.2 信息封闭的趋势

以 GPT-4 技术报告为例——这篇论文几乎没有公开任何关于数据、架构、训练细节的信息。这与早期论文（如 GPT-2、BERT）详细公开所有细节形成了鲜明对比。

**这意味着什么？** 前沿模型的构建知识越来越集中在少数公司手中，外界研究者如果不主动学习底层技术，将彻底被排除在这一领域之外。

---

## 3. "更大即不同" 与可迁移的知识

### 3.1 规模带来的质变

课程引用了两个关键例子说明小模型和大模型之间存在质的差异：

**例子 1：计算分配的变化**
- 在小模型中，Attention 和 MLP 的 FLOPs 占比相对均衡
- 随着模型变大，MLP 消耗的 FLOPs 比例显著增加
- 这意味着小模型上的架构优化经验不一定适用于大模型

**例子 2：能力涌现 (Emergence)**
- 某些能力（如多步推理、算术）在模型规模达到某个阈值后"突然"出现
- 小模型上完全看不到这些能力的踪迹
- 这使得在小规模上预测大规模行为变得非常困难

### 3.2 三类可迁移的知识

尽管规模带来差异，课程仍然认为有三类知识值得学习：

| 知识类型 | 定义 | 是否可迁移 | 举例 |
|----------|------|------------|------|
| **Mechanics (机制)** | 事物如何工作 | ✅ 完全可迁移 | Transformer 架构、模型并行的原理 |
| **Mindset (思维方式)** | 如何最大化利用硬件 | ✅ 完全可迁移 | 效率优先、用 scaling laws 指导决策 |
| **Intuitions (直觉)** | 什么样的数据/模型决策带来好结果 | ⚠️ 部分可迁移 | 激活函数选择、超参数偏好 |

### 3.3 关于直觉的不确定性

课程引用了 Noam Shazeer 2020 年引入 SwiGLU 激活函数的论文。SwiGLU 在实验中效果很好，但论文中关于"为什么"的解释是：

> "We offer no explanation as to why these architectures seem to work; we attribute their success to divine benevolence."
> （我们无法解释为什么这些架构有效；我们将其成功归因于神圣的恩赐。）

这说明了 LLM 领域的一个现实：很多设计决策目前还无法从理论上证明，只能通过大量实验来验证。

---

## 4. 苦涩的教训 (The Bitter Lesson)

### 4.1 正确的理解

Rich Sutton 的 "The Bitter Lesson" 常被误解为"规模是唯一重要的东西"。课程给出了更精准的解读：

- ❌ **错误理解**：规模是唯一重要的，算法不重要
- ✅ **正确理解**：**能够随规模增长的算法**才是重要的

### 4.2 效率公式

课程提出了一个核心等式：

```
accuracy = efficiency × resources
```

关键洞察：**效率在大规模下比小规模更重要**。原因很简单——浪费 1% 的效率，如果你的训练预算是 $1000 那只浪费 $10；但如果预算是 $100M，那就是 $1M 的浪费。

引用的数据：2012-2019 年间 ImageNet 上的算法效率提升了 **44 倍**，这意味着通过更好的算法，你可以用 1/44 的计算量达到相同的准确率。

### 4.3 课程的核心问题框架

> 给定固定的计算和数据预算，能构建的最好模型是什么？
> 换句话说：**最大化效率！**

这个框架贯穿整个课程的每一个模块。

---

## 5. 语言模型发展全景

### 5.1 前神经网络时代 (2010年以前)

| 里程碑 | 年份 | 核心贡献 |
|--------|------|----------|
| Shannon 信息论 | 1950 | 用语言模型测量英语的熵，奠定了信息论基础。Shannon 估计英语的每字符熵约为 1.0-1.5 bits |
| N-gram 语言模型 | ~2007 | Google 的 Brants et al. 在 2T tokens 上训练 5-gram 模型用于机器翻译。统计方法的巅峰 |

### 5.2 神经网络关键组件 (2010s)

这一时期奠定了现代 LLM 的所有核心技术：

| 组件 | 论文 | 年份 | 核心思想 |
|------|------|------|----------|
| 神经语言模型 | Bengio et al. | 2003 | 用前馈神经网络在最后 n 个词上预测下一个词。首次将神经网络用于语言建模，引入了词嵌入的概念 |
| Seq2Seq | Sutskever et al. | 2014 | 将整个句子编码为单个向量，再解码翻译。Encoder-Decoder 架构的起源 |
| Adam 优化器 | Kingma & Ba | 2014 | 结合 RMSProp 和 Momentum 的自适应学习率优化器。至今仍是最广泛使用的优化器之一 |
| Attention 机制 | Bahdanau et al. | 2015 | 解决了 Seq2Seq 的瓶颈——不再需要将所有信息压缩到单个向量中。解码器可以"关注"输入序列的不同部分 |
| Transformer | Vaswani et al. | 2017 | "Attention Is All You Need"——完全基于 attention 的架构，抛弃了 RNN。引入了 multi-head attention、positional encoding。**现代 LLM 的基石** |
| Mixture of Experts | Shazeer et al. | 2017 | 条件计算——每个 token 只激活部分参数，实现参数量和计算量的解耦 |
| 模型并行 | GPipe/ZeRO/Megatron | 2018-19 | 突破单 GPU 内存限制，将模型分布到多个 GPU 上训练。ZeRO 可在 400 GPU 上训练 100B 参数模型 |

### 5.3 早期基础模型 (2018-2020)

| 模型 | 组织 | 参数量 | 核心创新 |
|------|------|--------|----------|
| ELMo | AllenAI | - | 用 LSTM 预训练，证明了预训练+微调范式的有效性 |
| BERT | Google | 340M | 用 Transformer 预训练 (Masked LM)，在几乎所有 NLP 任务上刷新了 SOTA |
| GPT-2 | OpenAI | 1.5B | 首次展示流畅文本生成和零样本能力的潜力。开创了"阶段性发布" (staged release) |
| T5 | Google | 11B | 将所有 NLP 任务统一为 text-to-text 格式。引入了 C4 (Colossal Cleaned Common Crawl) 数据集 |

### 5.4 拥抱规模，走向封闭 (2020-2022)

| 模型 | 组织 | 参数量 | 训练数据 | 关键点 |
|------|------|--------|----------|--------|
| GPT-3 | OpenAI | 175B | 300B tokens | 展示了 in-context learning (少样本学习)。**首次闭源** |
| Scaling Laws | OpenAI (Kaplan) | - | - | 发现了模型大小、数据量、计算量之间的幂律关系。"更大的模型需要更少的 token" |
| PaLM | Google | 540B | - | 大规模训练 (6144 TPUv4)，但**欠训练** (数据量不够) |
| Chinchilla | DeepMind | 70B | 1.4T tokens | **计算最优 scaling laws**：模型和数据应以相同速率扩大。D* = 20N* |

**Chinchilla 的核心发现：** PaLM 540B 用了比 Chinchilla 更多的计算量，但因为数据不够，效果反而更差。Chinchilla 70B 在相同计算预算下通过使用更多数据击败了更大的模型。这彻底改变了训练策略——不再盲目追求大模型，而是追求计算最优的模型大小和数据量配比。

### 5.5 开源模型浪潮

| 模型 | 组织 | 参数量 | 特点 |
|------|------|--------|------|
| The Pile + GPT-J | EleutherAI | 6.7B | 开放数据集 (825GB, 22个子集) + 开放模型 |
| OPT | Meta | 175B | GPT-3 复现尝试，论文详细记录了大量硬件故障 (992 A100, 2个月) |
| BLOOM | BigScience/HF | 176B | 聚焦多语言数据来源 (48×8 A100, 3.5个月) |
| LLaMA 1/2/3 | Meta | 7B-405B | 里程碑式的开放模型系列。LLaMA1 仅用开放数据训练，架构: Pre-norm + SwiGLU + RoPE |
| Qwen 2.5 | 阿里巴巴 | - | 中国领先的开放模型系列 |
| DeepSeek 67B/V2/V3 | DeepSeek | 67B-671B | 引入 MLA (Multi-head Latent Attention)，V3 是 MoE 架构 |
| OLMo 2 | AI2 | 7B-32B | 完全开源 (权重 + 数据 + 代码 + 训练日志) |

### 5.6 开放程度的三个层次

```
封闭模型 (Closed)         → 仅 API 访问 (如 GPT-4o)
                              不公开任何内部细节

开放权重 (Open-weight)     → 权重可下载，论文有架构细节
                              部分训练细节，无数据细节 (如 DeepSeek V3)

开源模型 (Open-source)     → 权重 + 数据 + 代码 + 训练日志全部公开
                              但不一定包含设计理由和失败实验 (如 OLMo)
```

---

## 6. 课程五大模块详解

### 核心框架：一切关乎效率

```
资源 = 数据 + 硬件 (计算, 内存, 通信带宽)
目标 = 在固定资源下训练最好的模型
```

具体场景：给你一个 Common Crawl 数据集和 32 张 H100 运行 2 周，你该怎么做？

### 模块 1: 基础 (Basics) — Assignment 1

**目标**：搭建完整的基础 pipeline

三大组件：
- **分词 (Tokenization)**：字符串 ↔ 整数序列的转换 (详见第7节)
- **模型架构 (Architecture)**：Transformer 及其变体
- **训练 (Training)**：优化器、学习率调度、正则化

**架构变体汇总**：

| 组件 | 经典方案 | 现代改进 | 动机 |
|------|----------|----------|------|
| 激活函数 | ReLU | SwiGLU | 实验效果更好 (无理论解释) |
| 位置编码 | Sinusoidal | RoPE | 可外推到更长序列，编码相对位置 |
| 归一化 | LayerNorm | RMSNorm | 移除均值计算，减少计算量 |
| 归一化位置 | Post-Norm | Pre-Norm | 训练更稳定 |
| MLP | Dense | MoE | 参数量与计算量解耦 |
| Attention | Full | Sliding Window / Linear | 降低计算复杂度 O(n²) → O(n) |
| KV Heads | Multi-Head | GQA / MLA | 减少 KV cache 内存，加速推理 |

**训练组件详解**：

| 组件 | 选项 | 说明 |
|------|------|------|
| 优化器 | AdamW, Muon, SOAP | AdamW 是标准选择；Muon 和 SOAP 是新兴替代方案 |
| 学习率调度 | Cosine, WSD | Cosine 是经典选择；WSD (Warmup-Stable-Decay) 来自清华 MiniCPM |
| Batch Size | 逐步增大 | "Critical batch size" 概念：超过此值，增加 batch size 不再提高样本效率 |
| 正则化 | Dropout, Weight Decay | 现代大模型趋向于不使用 Dropout |

### 模块 2: 系统 (Systems) — Assignment 2

**目标**：最大化硬件利用率

**GPU 架构理解**：
- A100 理论算力：312 TFLOP/s (BF16)
- H100 理论算力：~990 TFLOP/s (BF16, 非稀疏)
- 关键瓶颈不是计算而是**数据移动**
- 类比：DRAM 是仓库 (大但远)，SRAM 是工厂 (小但快)

**三大系统主题**：

#### Kernel 编程
- 目标：减少 GPU 内部的数据移动
- 工具：CUDA / **Triton** / CUTLASS / ThunderKittens
- 核心思想：将尽可能多的计算放在 SRAM 中完成，减少 DRAM 访问

#### 并行策略
- GPU 间数据移动比 GPU 内更慢，但原则相同
- 需要分片 (shard) 的对象：参数、激活值、梯度、优化器状态
- 四种并行策略：

| 策略 | 分片维度 | 说明 |
|------|----------|------|
| Data Parallelism | 数据 (batch) | 每个 GPU 持有完整模型，处理不同数据 |
| Tensor Parallelism | 模型层内 | 将矩阵乘法分拆到多个 GPU |
| Pipeline Parallelism | 模型层间 | 不同层放在不同 GPU 上 |
| Sequence Parallelism | 序列长度 | 长序列分拆到多个 GPU |

#### 推理优化
- **两个阶段**：
  - Prefill（预填充）：处理 prompt，所有 token 可并行，**计算密集型**
  - Decode（解码）：逐 token 生成，**内存密集型** (受带宽限制)
- **加速方法**：
  - 模型压缩：剪枝、量化、蒸馏
  - 投机解码 (Speculative Decoding)：用小模型批量生成候选 token，大模型并行验证 (保证精确解码！)
  - 系统优化：KV 缓存、请求批处理

### 模块 3: 缩放定律 (Scaling Laws) — Assignment 3

**核心问题**：给定计算预算 C，应该用更大的模型 N 还是更多的数据 D？

**两个关键论文**：
1. **Kaplan (OpenAI, 2020)**：发现幂律关系，"更大的模型需要更少的 token"
2. **Chinchilla (DeepMind, 2022)**：修正了 Kaplan 的结论

**Chinchilla 方法论**（三种方法）：
- 方法1：固定模型大小，用4个学习率训练，变化 token 数量，拟合下包络线
- 方法2 (IsoFLOP)：固定 FLOPs 预算，变化模型大小，取最优点
- 方法3：拟合参数化函数 `L(N,D) = E + A/N^α + B/D^β`

**核心结论**：`D* = 20 × N*`
- 1.4B 参数模型应在 28B tokens 上训练
- 模型和数据应以相同速率扩大

**但有一个重要限制**：Chinchilla 只优化训练成本，不考虑推理成本。在实际生产中，推理计算量远超训练计算量，因此可能需要**过度训练 (overtrain)** 较小的模型。

### 模块 4: 数据 (Data) — Assignment 4

**核心问题**：我们想让模型具备什么能力？多语言？代码？数学？

**数据来源的多样性** (以 The Pile 为例)：
- CommonCrawl 网页、PubMed 医学文献、ArXiv 论文
- GitHub 代码、StackExchange 问答、USPTO 专利
- 维基百科、书籍、新闻等
- 共 825GB 文本，22 个子集

**数据处理 Pipeline**：

```
原始数据 (HTML/PDF)
    ↓ 转换 (Transformation)
纯文本 / Markdown
    ↓ 过滤 (Filtering)
高质量数据
    ↓ 去重 (Deduplication)
最终训练数据
```

- **转换**：HTML → Markdown/文本，保留内容和部分结构
- **过滤**：质量分类器 (保留高质量)、有害内容分类器 (移除有害内容)
- **去重**：节省计算、避免记忆化。技术：Bloom Filter、MinHash

**实际网页数据的现实**：课程代码中实际下载了 Common Crawl 数据并进行了查看——"It's a wasteland out there!" (外面是一片废墟！) 说明原始网页数据质量极差，需要大量处理。

**法律问题**：
- 是否可以援引"合理使用"来训练版权数据？仍在争议中
- Google 与 Reddit 签订了数据许可协议

### 模块 5: 对齐 (Alignment) — Assignment 5

**从基础模型到实用模型**：

基础模型 (Base Model) 是"原始潜力"——擅长续写文本，但不一定会遵循指令。对齐使模型真正有用。

**对齐的三个目标**：
1. 遵循指令
2. 调整风格 (格式、长度、语气)
3. 安全性 (拒绝回答有害问题)

**阶段 1: 有监督微调 (SFT)**

```python
# 数据格式
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 1 + 1?"},
    {"role": "assistant", "content": "The answer is 2."},
]
```

- 用 (prompt, response) 对进行监督学习
- LIMA 论文的关键洞察：基础模型已经具备能力，SFT 只需少量高质量数据来"唤醒"这些能力
- 目标函数：最大化 `p(response | prompt)`

**阶段 2: 从反馈中学习 (Learning from Feedback)**

偏好数据格式：
```
Prompt: "训练语言模型的最佳方式是什么？"
Response A: "你应该使用大数据集训练很长时间。"
Response B: "你应该使用小数据集训练很短时间。"
Chosen: A
```

**验证器 (Verifiers)**：
- 形式验证器：代码执行、数学证明 (可精确判断对错)
- 学习的验证器：用 LM-as-a-judge

**三大算法**：

| 算法 | 来源 | 核心思想 |
|------|------|----------|
| PPO | OpenAI (2017/2022 InstructGPT) | 强化学习方法，需要训练 reward model 和 value function |
| DPO | Stanford (2023) | 直接从偏好数据优化策略，无需 reward model，更简单 |
| GRPO | DeepSeek (2024) | 在 DPO 基础上移除 value function，进一步简化 |

### 效率如何驱动设计决策

课程最后总结了一个核心洞察——当前我们处于**计算受限** (compute-constrained) 的时代：

| 设计决策 | 效率驱动的理由 |
|----------|----------------|
| 数据处理 | 避免在低质量数据上浪费计算 |
| 分词 | 直接处理 bytes 很优雅，但在当前架构下计算效率低 |
| 模型架构 | 很多改动是为了减少内存或 FLOPs (如 GQA 共享 KV cache) |
| 训练 | 数据足够时可以只训练一个 epoch！ |
| 缩放定律 | 用小模型低成本实验来预测大模型的超参数 |
| 对齐 | 更好的对齐 = 可以用更小的基础模型达到相同效果 |

> 课程预言：未来我们将从**计算受限**转变为**数据受限**...

---

## 7. 分词 (Tokenization) 深入剖析

> 灵感来源：Andrej Karpathy 的分词教学视频

### 7.1 为什么需要分词？

语言模型的输入是**整数序列** (tokens)，而人类使用的是**Unicode 字符串**。分词器 (Tokenizer) 就是在这两者之间转换的桥梁。

```python
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:     # 字符串 → token 序列
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:     # token 序列 → 字符串
        raise NotImplementedError
```

关键要求：
- `decode(encode(string)) == string` (必须能完美往返转换)
- **词表大小** (vocabulary size)：可能的 token 数量
- **压缩率** (compression ratio)：`原始字节数 / token 数`，越高越好

### 7.2 方案 1：字符级分词 (Character Tokenizer)

**原理**：将每个 Unicode 字符映射为其 code point (码点)。

```python
ord("a") == 97
ord("🌍") == 127757
chr(97) == "a"
chr(127757) == "🌍"
```

**实现**：
```python
class CharacterTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
```

**优缺点分析**：

| 维度 | 评价 |
|------|------|
| 词表大小 | ❌ ~150K Unicode 字符，太大 |
| 压缩率 | ❌ 每个字符一个 token，对中英文还算可以，但不够高效 |
| 稀有字符 | ❌ 很多字符 (如 emoji) 极其罕见，浪费词表空间 |
| 实现简单度 | ✅ 极其简单 |

### 7.3 方案 2：字节级分词 (Byte Tokenizer)

**原理**：将字符串编码为 UTF-8 字节序列，每个字节 (0-255) 作为一个 token。

UTF-8 编码的特点：
```python
bytes("a", encoding="utf-8")  == b"a"           # ASCII: 1 字节
bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"  # emoji: 4 字节
bytes("你", encoding="utf-8") == b"\xe4\xbd\xa0"       # 中文: 3 字节
```

**实现**：
```python
class ByteTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        return list(map(int, string.encode("utf-8")))
    def decode(self, indices: list[int]) -> str:
        return bytes(indices).decode("utf-8")
```

**优缺点分析**：

| 维度 | 评价 |
|------|------|
| 词表大小 | ✅ 仅 256，非常紧凑 |
| 压缩率 | ❌ 恰好为 1.0 (每字节一个 token)，terrible！ |
| 序列长度 | ❌ 序列太长，Transformer attention 是 O(n²) 的 |
| 通用性 | ✅ 可以表示任何字符串 |

**为什么压缩率是 1.0 很糟糕？**
- 一个英文单词 "hello" 就需要 5 个 token
- 中文 "你好" 需要 6 个 token (每个汉字 3 字节)
- Transformer 的上下文窗口有限（如 2048 或 4096），这意味着能处理的实际内容大大减少

### 7.4 方案 3：词级分词 (Word Tokenizer)

**原理**：用正则表达式将字符串拆分为"词"。

简单版：
```python
segments = regex.findall(r"\w+|.", "I'll say supercalifragilisticexpialidocious!")
# → ["I", "'", "ll", " ", "say", " ", "supercalifragilisticexpialidocious", "!"]
```

GPT-2 使用的更复杂的正则：
```python
GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

这个正则的含义：
- `'(?:[sdmt]|ll|ve|re)` — 英语缩写后缀 ('s, 'd, 'm, 't, 'll, 've, 're)
- `?\p{L}+` — 可选空格 + Unicode 字母序列
- `?\p{N}+` — 可选空格 + Unicode 数字序列
- `?\[^\s\p{L}\p{N}]+` — 可选空格 + 非空白非字母非数字字符
- `\s+(?!\S)` — 尾部空白
- `\s+` — 其他空白

**优缺点分析**：

| 维度 | 评价 |
|------|------|
| 词表大小 | ❌ 巨大且不固定 (等于语料中的不同词数) |
| 压缩率 | ✅ 很高 (常见词一个 token) |
| OOV 问题 | ❌ 训练时未见过的词只能用 UNK token，丢失信息 |
| 稀有词 | ❌ 长尾分布，大量罕见词浪费词表 |

### 7.5 方案 4：BPE 分词 (Byte Pair Encoding) ⭐

**历史**：
1. 1994 年 Philip Gage 发明 BPE 用于**数据压缩**
2. 2016 年 Sennrich et al. 将其应用于**神经机器翻译**的子词分割
3. 2019 年 GPT-2 使其成为 LLM 的标准分词方法

**核心思想**：
- 从字节 (256 个基础 token) 开始
- 反复合并语料中最频繁出现的**相邻 token 对**
- 自动学习出一个大小可控的词表

**训练算法详解**：

```
输入: string = "the cat in the hat", num_merges = 3

Step 0: 转为字节序列
indices = [116, 104, 101, 32, 99, 97, 116, 32, 105, 110, 32, 116, 104, 101, 32, 104, 97, 116]
          [ t    h    e   ' '  c   a    t   ' '  i    n   ' '  t    h    e   ' '  h    a    t ]

Step 1: 统计所有相邻 token 对的频次
  (116, 104) → 2  # "th" 出现 2 次
  (104, 101) → 2  # "he" 出现 2 次
  (101, 32)  → 2  # "e " 出现 2 次
  ...
  最频繁: (116, 104) = "th" → 合并为 token 256
  indices = [256, 101, 32, 99, 97, 116, 32, 105, 110, 32, 256, 101, 32, 104, 97, 116]

Step 2: 重新统计
  (256, 101) → 2  # "the" 出现 2 次
  ...
  最频繁: (256, 101) = "the" → 合并为 token 257
  indices = [257, 32, 99, 97, 116, 32, 105, 110, 32, 257, 32, 104, 97, 116]

Step 3: 重新统计
  (257, 32)  → 2  # "the " 出现 2 次
  ...
  最频繁: (257, 32) = "the " → 合并为 token 258
  indices = [258, 99, 97, 116, 32, 105, 110, 32, 258, 104, 97, 116]
```

**合并操作的实现**：

```python
def merge(indices, pair, new_index):
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)  # 合并
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices
```

**编码过程** (使用训练好的分词器)：

对新文本编码时，按训练时的合并顺序依次应用所有合并规则：

```python
def encode(self, string):
    indices = list(map(int, string.encode("utf-8")))  # 先转为字节
    for pair, new_index in self.params.merges.items():  # 按顺序应用每个合并
        indices = merge(indices, pair, new_index)
    return indices
```

**解码过程**：

```python
def decode(self, indices):
    bytes_list = [self.params.vocab[idx] for idx in indices]  # 每个 token → 字节序列
    return b"".join(bytes_list).decode("utf-8")               # 拼接并转为字符串
```

**BPE 的关键数据结构**：

```python
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]              # token ID → 字节表示
    merges: dict[tuple[int, int], int]   # (token1, token2) → 合并后的 token ID
```

**BPE 综合评价**：

| 维度 | 评价 |
|------|------|
| 词表大小 | ✅ 可控 (= 256 + num_merges) |
| 压缩率 | ✅ 常见词/子词被有效压缩 |
| OOV 问题 | ✅ 不存在！任何字符串都可以回退到字节级表示 |
| 适应性 | ✅ 自动适应训练语料的分布 |
| 性能 | ⚠️ 朴素实现 O(num_merges × seq_len)，需要优化 |

### 7.6 GPT-2 分词器实战观察

使用 OpenAI 的 `tiktoken` 库：

```python
tokenizer = tiktoken.get_encoding("gpt2")
indices = tokenizer.encode("Hello, 🌍! 你好!")
# 编码后的 token 序列
# decode(encode(x)) == x  ✅ 完美往返
```

**重要观察**：
1. 单词和前面的空格通常合为一个 token (如 `" world"`)
2. 同一个单词在句首和句中可能编码不同 (如 `"hello"` vs `" hello"`)
3. 数字通常按几位一组分割

### 7.7 分词的未来：无分词方案

课程提到了几个试图绕过分词的研究方向：

| 方法 | 论文 | 思路 |
|------|------|------|
| ByT5 | Google (2021) | 直接在字节上训练 T5 |
| MegaByte | Meta (2023) | 分层架构处理字节序列 |
| BLT | Meta (2024) | Byte Latent Transformer，字节到潜在表示 |
| Token-Free | 2024 | 无 token 的语言模型 |

这些方法"有前景但尚未在前沿规模上得到验证"。核心问题是字节序列太长，在当前 Transformer 架构下计算效率不足。

### 7.8 Assignment 1 中的分词挑战

课程要求学生在作业中实现以下改进：
1. **优化 encode()**：不遍历所有 merge，只应用相关的 merge
2. **特殊 token 处理**：检测并保留 `<|endoftext|>` 等特殊标记
3. **Pre-tokenization**：使用 GPT-2 正则表达式先将文本分段
4. **性能优化**：尽可能提高实现速度

---

## 8. 关键论文索引

### 基础理论
| 论文 | 年份 | 一句话总结 |
|------|------|------------|
| Shannon - Prediction and Entropy of Printed English | 1950 | 信息论奠基，语言熵的测量 |
| Bengio - A Neural Probabilistic Language Model | 2003 | 首个神经语言模型，引入词嵌入 |
| Vaswani - Attention Is All You Need | 2017 | Transformer 架构，现代 LLM 的基石 |

### 优化与训练
| 论文 | 年份 | 一句话总结 |
|------|------|------------|
| Kingma & Ba - Adam | 2014 | 自适应学习率优化器 |
| Loshchilov - AdamW | 2017 | 解耦权重衰减，修正 Adam |
| Loshchilov - Cosine LR | 2017 | 余弦学习率调度 |
| McCandlish - Large Batch Training | 2018 | 临界 batch size 概念 |
| MiniCPM/WSD | 2024 | Warmup-Stable-Decay 学习率调度 |

### 架构创新
| 论文 | 年份 | 一句话总结 |
|------|------|------------|
| Shazeer - SwiGLU | 2020 | 门控激活函数，实验效果好 |
| Su - RoPE | 2021 | 旋转位置编码，支持序列外推 |
| Ba - LayerNorm | 2016 | 层归一化 |
| Zhang - RMSNorm | 2019 | 只用均方根的简化归一化 |
| Ainslie - GQA | 2023 | 分组查询注意力，减少 KV cache |
| DeepSeek - MLA | 2024 | 多头潜在注意力 |

### 缩放定律
| 论文 | 年份 | 一句话总结 |
|------|------|------------|
| Kaplan - Scaling Laws | 2020 | 发现模型/数据/计算的幂律关系 |
| Hoffmann - Chinchilla | 2022 | 计算最优训练，D* = 20N* |

### 对齐
| 论文 | 年份 | 一句话总结 |
|------|------|------------|
| Ouyang - InstructGPT | 2022 | RLHF 流程：SFT → RM → PPO |
| Rafailov - DPO | 2023 | 直接从偏好优化，简化 RLHF |
| DeepSeek - GRPO | 2024 | 移除 value function 的偏好优化 |
| Zhou - LIMA | 2023 | 少量高质量数据足以对齐 |

---

## 总结

Lecture 01 涵盖了三大核心内容：

1. **课程定位**：通过从零构建来理解 LLM，聚焦效率最大化
2. **领域全景**：从 Shannon 1950 到 DeepSeek R1 2025 的完整演化
3. **分词技术**：从 Character → Byte → Word → BPE 的逐步优化

核心公式：**accuracy = efficiency × resources**

下一讲：PyTorch 构建模块与资源核算 (Resource Accounting)
