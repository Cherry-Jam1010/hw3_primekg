# Mental Health Knowledge Graph Analysis System
# 基于 PrimeKG 的精神障碍知识图谱分析与可信问答系统

## 项目概述

本项目基于 PrimeKG（Precision Medicine Knowledge Graph）构建一个面向心理学与精神健康领域的"精神障碍知识图谱分析与可信问答网页系统"。系统包含四大核心模块：知识图谱分析、图嵌入模型训练、LLM幻觉检测、RAG增强问答，以及一个四页Web应用原型。

---

## 技术架构

```
mental_health_kg/
├── data/                    # 数据目录
│   ├── primekg.csv         # PrimeKG 完整数据（需下载）
│   └── mental_subgraph.csv  # 精神障碍子图
├── analysis/                # 图分析模块
│   ├── subgraph_extraction.py       # 子图提取
│   ├── graph_statistics.py          # 图统计
│   └── comorbidity_analysis.py      # 共病分析
├── models/                  # 图嵌入模型
│   ├── train_transE.py             # TransE 训练
│   ├── train_rotateE.py            # RotatE 训练
│   └── evaluation.py               # 模型评估
├── hallucination/           # 幻觉检测
│   ├── generate_triplets.py        # 三元组生成
│   ├── llm_judgment.py            # LLM判断
│   └── eval_hallucination.py       # 幻觉评估
├── rag/                     # RAG模块
│   ├── vector_store.py             # 向量存储
│   └── rag_chain.py                # RAG链
├── web/                     # Web应用
│   ├── app.py                       # 主应用
│   ├── routes/                      # 路由
│   └── templates/                   # 模板
└── results/                 # 结果存储
```

---

## 核心技术与算法

### 1. 知识图谱子图提取

**算法**：基于关键词匹配的子图采样
- 识别精神障碍节点：抑郁症(Depression)、焦虑障碍(Anxiety)、PTSD、精神分裂症(Schizophrenia)等
- 提取1-2跳邻居节点：症状、药物、表型
- 使用NetworkX构建子图

```python
# 关键词列表（心理学认定的精神障碍）
MENTAL_DISORDERS = [
    'depression', 'anxiety', 'ptsd', 'schizophrenia', 'bipolar',
    'ocd', 'autism', 'adhd', 'eating disorder', 'borderline',
    'panic', 'phobia', 'social anxiety', 'dementia', 'alzheimer',
    'personality disorder', 'dissociative', 'somatoform', 'trauma'
]
```

### 2. 图结构分析

**分析维度**：
- **本体结构**：节点类型分布、关系类型分布
- **核心节点识别**：度中心性、PageRank
- **共病模式**：同时与某症状相关的疾病对、共病率计算

### 3. TransE 知识图谱嵌入

**算法原理**：
- 将关系建模为头实体到尾实体的平移向量
- 目标函数：h + r ≈ t
- 损失函数：margin-based ranking loss

```
score(h, r, t) = ||h + r - t||_2^2
```

### 4. RotatE 知识图谱嵌入

**算法原理**：
- 将关系建模为复数空间中的旋转
- 头实体到尾实体的变换：h ∘ r = t
- 支持对称、反对称、逆关系

```
score(h, r, t) = -||h ∘ r - t||_2^2
```

### 5. 链接预测评估指标

- **MRR (Mean Reciprocal Rank)**：平均倒数排名
- **Hits@K**：排名前K中的正确实体比例
  - Hits@1: 精确匹配率
  - Hits@3: 前三命中率
  - Hits@10: Top-10准确率

### 6. 疾病大类预测

**算法**：基于图嵌入的节点分类
- 使用训练好的KGE（TransE/RotatE）提取疾病节点嵌入
- 应用分类器（Random Forest/SVM）预测疾病大类
- 评估分类准确率

### 7. LLM幻觉检测

**任务设计**：
- **真实三元组**：从PrimeKG随机抽取，改写为自然语言判断题
- **伪造三元组**：替换头实体/关系/尾实体生成
- **评估指标**：准确率、精确率、召回率、F1

**检测维度**：
- 疾病-症状关系
- 疾病-药物关系
- 疾病-表型关系
- 疾病-疾病关系

### 8. RAG增强问答

**架构**：
```
用户问题 → 向量检索 → Top-K相关块 → LLM生成 → 带证据的回答
```

**技术**：
- 向量数据库：ChromaDB（本地部署）
- Embedding模型：sentence-transformers
- LLM：OpenAI GPT-3.5/GPT-4 或本地模型

**评估维度**：
- 回答准确性
- 证据可追溯性
- 幻觉率对比

---

## 心理学知识映射

### 精神障碍分类（DSM-5/ICD-11）

| 大类 | 代表疾病 | PrimeKG关键词 |
|------|---------|--------------|
| 情感障碍 | 抑郁症、双相障碍 | depression, bipolar |
| 焦虑障碍 | 焦虑症、PTSD、惊恐障碍 | anxiety, ptsd, panic |
| 精神分裂症谱系 | 精神分裂症 | schizophrenia |
| 神经发育障碍 | 自闭症、ADHD | autism, adhd |
| 强迫及相关障碍 | 强迫症 | ocd |
| 创伤相关障碍 | 创伤后应激障碍 | trauma, stress |
| 饮食障碍 | 神经性厌食、暴食症 | eating disorder |
| 人格障碍 | 边缘型人格障碍 | borderline, personality |

### 常见症状节点

- 情绪症状：情绪低落、焦虑、情绪波动
- 认知症状：注意力不集中、记忆力下降
- 行为症状：社交退缩、睡眠障碍
- 躯体症状：头痛、胃肠不适

### 常用治疗药物

- SSRIs：氟西汀、舍曲林、帕罗西汀
- SNRIs：文拉法辛、度洛西汀
- 非典型抗精神病药：奥氮平、利培酮
- 情感稳定剂：锂盐、丙戊酸盐

---

## Web应用页面设计

### 页面1：精神障碍知识图谱浏览页

**功能**：
- 搜索疾病名称
- 展示相关症状、药物、疾病类别
- 关系网络图可视化（Cytoscape.js）

**数据展示**：
- 疾病节点：中心位置，高亮显示
- 症状节点：绿色
- 药物节点：蓝色
- 关系类型：边标签

### 页面2：疾病知识问答页

**功能**：
- 自然语言问题输入
- RAG回答生成
- PrimeKG三元组证据展示

**示例问题**：
- "抑郁症可能关联哪些症状？"
- "哪些药物与焦虑障碍相关？"
- "精神分裂症有哪些表型特征？"

### 页面3：LLM幻觉检测页

**功能**：
- 展示真实/伪造三元组
- LLM真假判断结果
- 正确率统计

**检测维度**：
- 疾病-症状关系
- 疾病-药物关系
- 疾病-表型关系

### 页面4：模型实验结果页

**功能**：
- TransE vs RotatE 性能对比表
- MRR、Hits@K 雷达图
- 疾病大类预测准确率

---

## 依赖技术栈

### 数据处理
- pandas, numpy
- networkx (图分析)

### 知识图谱嵌入
- pykeen (TransE, RotatE)
- torch

### RAG系统
- chromadb (向量数据库)
- sentence-transformers (嵌入模型)

### LLM集成
- openai / anthropic API
- langchain

### Web框架
- Flask / FastAPI
- HTML/CSS/JavaScript
- Cytoscape.js (图可视化)

---

## 项目定位

- **知识学习**：心理学课程精神障碍知识学习工具
- **教学展示**：知识图谱、图嵌入、LLM幻觉检测和RAG效果展示
- **科研辅助**：知识图谱分析、科研数据可视化

**注意**：本项目定位于知识学习、教学展示和科研辅助，不用于真实临床诊断或治疗决策。
