# Lecture 14 深入详细学习笔记：数据处理细节

> CS336: Language Models From Scratch (Spring 2025) — Stanford
> 源文件: `lecture_14.py` (可执行讲义)

---

## 目录

1. [概述与框架](#1-概述与框架)
2. [过滤算法：KenLM](#2-过滤算法kenlm)
3. [过滤算法：fastText](#3-过滤算法fasttext)
4. [过滤算法：DSIR](#4-过滤算法dsir)
5. [过滤应用](#5-过滤应用)
6. [去重 (Deduplication)](#6-去重-deduplication)

---

## 1. 概述与框架

### 1.1 通用过滤框架

```
给定: 目标数据 T (小规模高质量) + 原始数据 R (大规模低质量)
目标: 找到 R 的子集 T' 使其类似于 T

要求:
  - 从 T 泛化 (T' ≠ T)
  - 极其快速 (R 可能是数十 TB)
```

### 1.2 三种实例化

| 方法 | 模型类型 | 评分函数 | 选择方式 |
|------|---------|---------|---------|
| **KenLM** | 生成模型 p_T(x) | perplexity | 低于阈值保留 |
| **fastText** | 判别分类器 | p(T\|x) | 高于阈值保留 |
| **DSIR** | 重要性权重 | p_T(x)/p_R(x) | 按权重重采样 |

---

## 2. 过滤算法：KenLM

### 2.1 N-gram 语言模型 + Kneser-Ney 平滑

```
最大似然估计:
  p(in | the cat) = count(the cat in) / count(the cat)

问题: 大多数 n-gram 计数为 0
解决: Kneser-Ney 平滑 — 利用低阶 n-gram 的概率
  p(in | the cat) 也依赖于 p(in | cat)
```

### 2.2 使用方式

```python
model = kenlm.Model("wikipedia/en.arpa.bin")

# 计算 log p(content) 和 perplexity
score = model.score(content)
perplexity = exp(-score / num_tokens)
```

**Perplexity 示例**：
| 文本 | Perplexity |
|------|-----------|
| Stanford 创校介绍 | 低 (类似 Wikipedia) |
| 课程评分说明 | 中 |
| "asdf asdf asdf..." | 高 |
| "the the the..." | 中 (常见词重复) |

### 2.3 CCNet 的使用

- 按段落计算 perplexity
- 按 perplexity 升序排列
- 保留前 1/3（最像 Wikipedia 的）
- 被 LLaMA 采用

---

## 3. 过滤算法：fastText

### 3.1 模型结构

```
词袋 (Bag of Words):
  问题: V × K 个参数 (V=词汇量, K=类别)

fastText: 词嵌入 + 线性头
  输入 tokens → Embedding(V, H) → 平均池化 → Linear(H, K) → softmax
  参数量: H × (V + K)  << V × K
```

### 3.2 N-gram 袋 + Hashing Trick

```python
bigrams = ["the cat", "cat in", "in the", "the hat"]
# 问题: bigram 数量可能无界
# 解决: Hashing trick
num_bins = 10_000_000  # 实际用 10M bins
hashed = [mmh3.hash(bigram) % num_bins for bigram in bigrams]
```

### 3.3 质量过滤的特殊情况

K = 2（好 vs 差）→ fastText 退化为**线性分类器**。

---

## 4. 过滤算法：DSIR

### 4.1 重要性采样 (Importance Sampling)

```
目标分布 p (想从这里采样)
提议分布 q (手头有这些样本)

1. 从 q 采样
2. 计算权重 w ∝ p/q
3. 按权重重采样 → 得到 p 的近似样本
```

### 4.2 DSIR 的做法

```
问题: 目标数据 D_p 太小，无法拟合好的生成模型

解决: 使用 hashed n-grams
  1. 将 n-grams hash 到固定 bins
  2. 在 hash 空间拟合 unigram 模型 (简单计数)
  3. 用 p_T(x)/p_R(x) 做重要性重采样
```

**对比 fastText**：
- 更原则性地捕获多样性
- 计算复杂度相似
- 实际效果略好于启发式分类

---

## 5. 过滤应用

### 5.1 语言识别

**工具**：fastText 语言识别模型（支持 176 种语言）

**挑战**：
- 短文本难以判断
- 低资源语言困难
- 可能误过滤方言
- 代码混用 (code-switching) 无明确定义

**实例**：Dolma 保留 p(English) ≥ 0.5 的页面

### 5.2 质量过滤

| 模型 | 正样本 | 负样本 |
|------|--------|--------|
| GPT-3 | Wikipedia, WebText, Books | CommonCrawl |
| LLaMA | Wikipedia 引用的页面 | CommonCrawl |
| phi-1 | GPT-4 标注的教育价值高的代码 | 一般代码 |
| DCLM | OpenHermes + ELI5 | RefinedWeb |

### 5.3 毒性过滤

Dolma 使用 Jigsaw 有毒评论数据集训练 2 个 fastText 分类器：
- **hate 分类器**：{unlabeled, obscene} vs 其余
- **NSFW 分类器**：{obscene} vs 其余

---

## 6. 去重 (Deduplication)

### 6.1 为什么去重？

- 训练更高效（更少 tokens）
- 避免记忆化（缓解版权、隐私问题）
- C4 中某产品描述重复 61,036 次！

### 6.2 设计空间

| 维度 | 选项 |
|------|------|
| 项 (Item) | 句子 / 段落 / 文档 |
| 匹配 | 精确 / 共同子项 / 共同子项比例 |
| 动作 | 全删 / 保留一个 |

### 6.3 精确去重

```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
# 按 hash 分组 → 每组保留一个
deduped = [next(group) for h, group in groupby(sorted(items, key=hash), key=hash)]
```

**C4 的做法**：3-sentence spans 精确匹配去重。

### 6.4 Bloom Filter

**近似集合成员检测**：

| 特性 | 说明 |
|------|------|
| 内存 | 极其节省 |
| 假阴性 | 不存在（说"不在"一定不在） |
| 假阳性 | 小概率（说"在"可能不在） |
| 可扩展 | 假阳性率随 hash 函数数量**指数下降** |

**参数权衡**：
```
m = bins 数, k = hash 函数数, n = 插入项数
假阳性率: f ≈ (1 - e^{-kn/m})^k
最优 k = ln(2) × m/n → f ≈ 0.5^k

Dolma 设置: 假阳性率 = 10^{-15}
```

### 6.5 Jaccard 相似度 + MinHash

**Jaccard 相似度**：J(A,B) = |A∩B| / |A∪B|

**MinHash**：一种随机 hash 函数 h，满足 Pr[h(A)=h(B)] = J(A,B)

```python
def minhash(S, seed):
    return min(mmh3.hash(x, seed) for x in S)
# 碰撞概率 = Jaccard 相似度
```

### 6.6 局部敏感哈希 (LSH)

**目标**：当 J(A,B) > 阈值时让 A 和 B 碰撞。

**方法**：n 个 hash 函数分成 b 个 band，每个 band 有 r 个 hash。

```
碰撞条件: 至少一个 band 的所有 r 个 hash 都匹配
碰撞概率: P = 1 - (1 - J^r)^b

增大 r → 阈值右移（更难匹配）
增大 b → 阈值左移（更容易匹配）
阈值 ≈ (1/b)^{1/r}
```

**实际设置**：n=9000, b=20, r=450 → 阈值 ≈ 0.993

---

## 总结

| 工具 | 用途 | 复杂度 |
|------|------|--------|
| KenLM | 生成模型过滤 | O(n) |
| fastText | 判别分类器过滤 | O(n) |
| DSIR | 重要性重采样 | O(n) |
| Bloom Filter | 精确去重 | O(n) |
| MinHash + LSH | 模糊去重 | O(n) |

**核心理念**：有了这些工具（机制），需要花时间与数据相处来建立直觉。

---

下一讲：Lecture 15 — RLHF/对齐
