# Lecture 13 深入详细学习笔记：训练数据

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_13.py` (可执行讲义)

---

## 目录

1. [数据的重要性与训练阶段](#1-数据的重要性与训练阶段)
2. [早期数据集：BERT, GPT-2](#2-早期数据集bert-gpt-2)
3. [Common Crawl 及其处理](#3-common-crawl-及其处理)
4. [主要预训练数据集演变](#4-主要预训练数据集演变)
5. [版权与法律问题](#5-版权与法律问题)
6. [后训练数据](#6-后训练数据)

---

## 1. 数据的重要性与训练阶段

### 1.1 数据是最重要的

> **Hot take**：数据是训练语言模型中最需要搞对的东西。

- 开源模型（如 Llama 3）公开了架构和训练细节，但**数据几乎不公开**
- 保密原因：(i) 竞争动态 (ii) 版权风险

### 1.2 训练阶段

```
1. Pre-training: 大量低质量数据 (网络文本)
2. Mid-training: 高质量数据增强能力
3. Post-training: 指令微调/RL

从大量 → 小量
从低质量 → 高质量
```

| 阶段 | 数据量 | 质量 | 输出 |
|------|--------|------|------|
| Pre-training | 万亿 tokens | 低-中 | Base model |
| Mid-training | 十亿 tokens | 中-高 | Base model (增强) |
| Post-training | 百万-千万 | 高 | Instruct/Chat model |

---

## 2. 早期数据集：BERT, GPT-2

### 2.1 BERT (2019)

- **BooksCorpus**：Smashwords 上 7K 本免费自出版书，985M 词
- **Wikipedia**：62M 篇文章，329 种语言
- 重要：使用**文档**而非句子作为序列

**Wikipedia 要点**：
- 任何人可编辑，管理员回退破坏
- 定期 dump（每几周）
- **数据投毒风险**：在 dump 前注入恶意编辑

### 2.2 GPT-2 WebText (2019)

```
来源: Reddit 帖子中 karma ≥ 3 的外链页面
规模: 800 万页面, 40GB 文本
开源复制: OpenWebTextCorpus
```

---

## 3. Common Crawl 及其处理

### 3.1 Common Crawl

- 非营利组织，2007 年成立
- 每月一次网络爬取，已有 ~100 次（2008-2025）
- 两种格式：WARC（原始 HTML）和 WET（转换为文本，有损）

### 3.2 CCNet (2019)

```
目标: 自动构建大规模高质量数据集
流程:
  1. 去重：去除重复段落
  2. 语言识别：fastText 分类器
  3. 质量过滤：KenLM 5-gram 模型（看起来像 Wikipedia 的保留）
```

### 3.3 C4 (Colossal Clean Crawled Corpus, 2019)

```
起始: 2019 年 4 月 Common Crawl 快照 (1.4T tokens)
手动规则过滤:
  - 行尾有标点且 ≥ 5 词
  - 页面 ≥ 3 句
  - 去除含"脏词"的页面
  - 去除含 '{' 的页面（排除代码）
  - 英语概率 0.99
结果: 806 GB, 156B tokens
```

---

## 4. 主要预训练数据集演变

### 4.1 时间线

| 数据集 | 年份 | 规模 | 特点 |
|--------|------|------|------|
| GPT-3 | 2020 | 400B tokens | 质量分类器过滤 CC |
| The Pile | 2021 | 275B tokens | 22 个高质量领域，草根社区 |
| MassiveText | 2021 | 10.5 TB | Gopher 使用，手动规则过滤 |
| LLaMA | 2022 | 1.2T tokens | CCNet + C4 + 多源 |
| RefinedWeb | 2023 | 5T tokens | "Web data is all you need" |
| Dolma | 2024 | 3T tokens | AI2 开源，多源 |
| DCLM | 2024 | 3.8T tokens | 模型过滤 (fastText) |
| Nemotron-CC | 2024 | 6.3T tokens | 分类器集成 + 合成改写 |

### 4.2 The Pile 重要子集

| 子集 | 描述 |
|------|------|
| Pile-CC | 用 WARC + jusText 处理（优于 WET） |
| PubMed Central | 500 万篇论文，NIH 强制公开 |
| arXiv | 1991 年以来的预印本（LaTeX） |
| Project Gutenberg | 75K 本公共领域书籍 |
| Books3 | 196K 本书（来自影子图书馆，已因版权下架） |
| StackExchange | Q&A 格式，接近指令微调 |
| GitHub/The Stack | 3.1 TB 代码，仅保留宽松许可 |

### 4.3 DCLM 的模型过滤

```
正样本 (200K):
  - OpenHermes-2.5 (GPT-4 生成的指令数据)
  - ELI5 (Reddit 科普问答)
负样本 (200K):
  - RefinedWeb
训练 fastText 分类器 → 过滤 240T tokens → 3.8T tokens
```

### 4.4 Nemotron-CC 的创新

- FineWebEdu 和 DCLM 过滤过于激进（去除 90% 数据）
- **分类器集成**：Nemotron-340B + DCLM 分类器
- **合成数据改写**：
  - 低质量数据 → LM 改写
  - 高质量数据 → LM 生成 QA 对

---

## 5. 版权与法律问题

### 5.1 版权法基础

```
版权保护: "原创作品固定在有形媒介中的表达"
保护对象: 表达 (expression), 不是想法 (idea)
注册费用: $65
有效期: 75 年
关键: 互联网上大多数内容都有版权！
```

### 5.2 合法使用版权作品的方式

1. **获取许可证** (License)
   - Creative Commons 许可
   - 商业许可（Google-Reddit, OpenAI-Shutterstock 等）

2. **主张合理使用 (Fair Use)** — 四个因素：
   - 使用目的和特征（教育 > 商业，变革性 > 复制性）
   - 版权作品的性质（事实性 > 虚构性）
   - 使用的数量和实质性
   - 对原作品市场的影响

### 5.3 对基础模型的考量

- 复制数据（训练的第一步）本身可能已经侵权
- 训练 ML 模型是**变革性**的（远不是复制粘贴）
- ML 系统关注**想法**而非具体**表达**
- 但：语言模型确实能影响市场（作家、艺术家）

---

## 6. 后训练数据

### 6.1 长上下文

- Transformer 与序列长度成二次关系 → 不适合从头预训练长上下文
- **LongLoRA**：将 Llama2 7B 从 4K 扩展到 100K tokens

### 6.2 任务型数据

- **Super-Natural Instructions**：1.6K+ 任务
- **FLAN 2022**：1.8K+ 任务（zero/few-shot + CoT 版本）

### 6.3 指令/对话数据

| 数据集 | 来源 | 规模 |
|--------|------|------|
| Alpaca | GPT-3.5 self-instruct | 52K |
| Vicuna | ShareGPT 对话 | 70K |
| Baize | GPT-3.5 self-chat | 111.5K |
| WizardLM | Evol-Instruct | 250K |
| OpenHermes 2.5 | GPT-4 多源聚合 | 1M |
| Llama 2 chat | 高质量人工标注 | 27.5K |

**趋势**：从人工标注 → 合成数据（GPT-4 生成）→ 开源模型生成（商业可行）

---

## 总结

| 要点 | 内容 |
|------|------|
| 数据不是天上掉的 | 需要大量工作获取和处理 |
| 流程 | Live service → 原始数据 → 处理后数据 |
| 数据是差异化关键 | 架构可以抄，数据不容易 |
| 法律伦理问题 | 版权、隐私是真实挑战 |
| 大量启发式 | 很多改进空间 |

---

下一讲：Lecture 14 — 数据处理细节
