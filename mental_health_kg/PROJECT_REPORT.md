# 精神障碍知识图谱分析与可信问答系统 - 技术报告

> 本报告详细说明系统的设计思路、技术实现、算法原理和使用方法

---

## 一、项目背景与目标

### 1.1 项目背景

随着大语言模型(LLM)在医疗健康领域的应用日益广泛，LLM产生的"幻觉"（即生成看似合理但实际错误的信息）成为一个严重问题。在精神健康领域，错误信息可能导致严重后果。

PrimeKG（Precision Medicine Knowledge Graph）是一个整合了20个高质量生物医学资源的知识图谱，包含17,080种疾病和超过405万条关系。本项目基于PrimeKG构建精神障碍知识图谱分析系统，用于：

1. 分析精神障碍子图的图结构特征
2. 训练知识图谱嵌入模型（TransE、RotatE）
3. 评估LLM在精神障碍知识上的幻觉倾向
4. 构建RAG增强的可靠问答系统

### 1.2 项目目标

- **知识学习**：作为心理学课程的精神障碍知识学习工具
- **教学展示**：展示知识图谱、图嵌入、LLM幻觉检测和RAG的工作原理
- **科研辅助**：提供可交互的知识图谱分析原型

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web 前端 (Flask + HTML/CSS/JS)           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ 图谱浏览 │  │ 知识问答 │  │ 幻觉检测 │  │ 实验结果 │      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
└────────┼─────────────┼─────────────┼─────────────┼────────────┘
         │             │             │             │
┌────────▼─────────────▼─────────────▼─────────────▼────────────┐
│                      Flask API Layer                            │
│   /api/search_disease  /api/answer_question                    │
│   /api/hallucination_test  /api/model_results                  │
└────────┬─────────────┬─────────────┬─────────────┬────────────┘
         │             │             │             │
┌────────▼─────────────▼─────────────▼─────────────▼────────────┐
│                    业务逻辑层 (Python Modules)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Graph    │  │ RAG      │  │Halluci- │  │ Model    │        │
│  │ Analysis │  │ Chain    │  │nation   │  │ Training │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└────────┬─────────────┬─────────────┬─────────────┬────────────┘
         │             │             │             │
┌────────▼─────────────▼─────────────▼─────────────▼────────────┐
│                      数据层                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ PrimeKG  │  │ 子图     │  │ 模型     │  │ 向量     │        │
│  │ 原始数据  │  │ CSV      │  │ 嵌入     │  │ 数据库   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
mental_health_kg/
├── analysis/                      # 图分析模块
│   ├── subgraph_extraction.py     # 子图提取
│   └── graph_statistics.py        # 统计分析与可视化
├── models/                        # 知识图谱嵌入模型
│   ├── train_transE.py            # TransE 模型训练
│   ├── train_rotateE.py           # RotatE 模型训练
│   ├── evaluation.py              # 模型评估与对比
│   └── disease_classification.py  # 疾病大类预测
├── hallucination/                 # 幻觉检测模块
│   ├── generate_triplets.py       # 三元组生成
│   └── eval_hallucination.py      # 幻觉评估
├── rag/                           # RAG 模块
│   ├── vector_store.py            # 向量数据库
│   └── rag_chain.py               # RAG 问答链
├── web/                           # Web 应用
│   ├── app.py                     # Flask 主程序
│   ├── templates/                 # HTML 模板 (4个页面)
│   └── static/css/                # 样式文件
├── results/                       # 结果输出
├── data/                          # 数据目录
├── main.py                        # 主入口
└── README.md                      # 项目说明
```

---

## 三、核心技术与算法

### 3.1 知识图谱子图提取

#### 3.1.1 关键词匹配策略

从PrimeKG中提取精神障碍子图，采用多轮关键词匹配：

```python
MENTAL_DISORDER_KEYWORDS = [
    # 情感障碍
    'depression', 'depressive', 'bipolar',
    # 焦虑障碍
    'anxiety', 'panic', 'phobia', 'agoraphobia',
    # 创伤相关
    'ptsd', 'trauma', 'stress disorder',
    # 精神分裂症谱系
    'schizophrenia', 'psychosis', 'delusion',
    # 神经发育障碍
    'autism', 'adhd', 'attention deficit',
    # ... 更多关键词
]
```

#### 3.1.2 子图采样算法

1. **疾病节点识别**：遍历PrimeKG中的所有节点，通过关键词匹配识别精神障碍节点
2. **邻居扩展**：提取1-2跳邻居节点（症状、药物、表型等）
3. **关系保留**：保留所有涉及精神障碍节点的边

### 3.2 图结构分析

#### 3.2.1 节点类型分布分析

统计子图中不同类型节点的数量和比例：
- disease（疾病）
- phenotype/symptom（表型/症状）
- drug（药物）
- gene（基因）
- pathway（通路）

#### 3.2.2 关系类型分布分析

分析边上的`display_relation`和`relation`字段，统计各关系类型的频率。

#### 3.2.3 中心性分析

使用NetworkX计算：
- **度中心性(Degree Centrality)**：节点的连接数
- **PageRank**：节点在图中的重要性
- **介数中心性(Betweenness Centrality)**：节点在路径中的桥梁作用

#### 3.2.4 共病模式分析

通过共享症状/表型来识别共病对：

```python
# 构建症状到疾病的映射
symptom_map = defaultdict(set)
for _, row in disease_phenotype_edges.iterrows():
    symptom_map[row['y_name']].add(row['x_name'])

# 计算共病对
for s1, s2 in combinations(symptoms, 2):
    shared = symptom_map[s1] & symptom_map[s2]
    if len(shared) >= 2:
        # 发现共病对
```

### 3.3 TransE 知识图谱嵌入

#### 3.3.1 算法原理

TransE是最经典的KGE模型，将关系建模为头实体到尾实体的"平移"：

```
h + r ≈ t
```

其中：
- `h`: 头实体嵌入向量
- `r`: 关系嵌入向量
- `t`: 尾实体嵌入向量
- `≈`: 使得 `h + r` 接近 `t`

#### 3.3.2 损失函数

TransE使用基于间隔的排序损失(Margin-based Ranking Loss)：

```
L = Σ max(0, γ + score(h,r,t) - score(h',r,t'))
```

其中：
- `(h,r,t)`: 正样本三元组
- `(h',r,t')`: 负样本三元组（通过替换头或尾实体生成）
- `γ`: 间隔超参数

#### 3.3.3 PyKEEN实现

```python
from pykeen.models import TransE
from pykeen.pipeline import pipeline

result = pipeline(
    model=TransE,
    model_kwargs={'embedding_dim': 200},
    training=training_factory,
    training_kwargs={'num_epochs': 200},
    evaluator=RankBasedEvaluator
)
```

### 3.4 RotatE 知识图谱嵌入

#### 3.4.1 算法原理

RotatE将关系建模为复数空间中的旋转：

```
h ∘ r = t
```

其中 `∘` 表示哈达玛积（元素级乘法）。关系被建模为复数平面上的旋转：

- 如果 `|h ∘ r| = |t|`，则关系成立
- RotatE能够自然地建模对称、反对称和逆关系

#### 3.4.2 优势

相比TransE，RotatE的优势：
1. **对称关系**：能够建模如"similar_to"这样的对称关系
2. **反对称关系**：能够建模如"parent_of"这样的反对称关系
3. **逆关系**：能够建模互为逆的关系

#### 3.4.3 PyKEEN实现

```python
from pykeen.models import RotatE

result = pipeline(
    model=RotatE,
    model_kwargs={
        'embedding_dim': 200,
        'interaction': 'complex_diagonal'
    },
    training=training_factory,
    training_kwargs={'num_epochs': 200},
    evaluator=RankBasedEvaluator
)
```

### 3.5 链接预测评估

#### 3.5.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **MRR** | 平均倒数排名 | Σ(1/rank_i) / N |
| **Hits@1** | 正确答案在前1名的比例 | #rank≤1 / N |
| **Hits@3** | 正确答案在前3名的比例 | #rank≤3 / N |
| **Hits@10** | 正确答案在前10名的比例 | #rank≤10 / N |

#### 3.5.2 评估过程

对于测试集中的每个三元组(h, r, t)：
1. 固定(h, r)，预测所有可能的尾实体
2. 按得分排序，得到尾实体t的排名rank_t
3. 固定(r, t)，预测所有可能的头实体
4. 按得分排序，得到头实体h的排名rank_h
5. 取两者的平均值或分别统计

### 3.6 疾病大类预测

#### 3.6.1 疾病分类体系

基于DSM-5/ICD-11分类标准：

| 类别ID | 名称 | 关键词 |
|--------|------|--------|
| depressive_disorders | 抑郁障碍 | depression, depressive |
| anxiety_disorders | 焦虑障碍 | anxiety, panic, phobia |
| trauma_stress_disorders | 创伤及应激相关障碍 | ptsd, trauma, stress |
| psychotic_disorders | 精神病性障碍 | schizophrenia, psychosis |
| bipolar_disorders | 双相及相关障碍 | bipolar, manic |
| neurodevelopmental_disorders | 神经发育障碍 | autism, adhd |
| obsessive_compulsive_disorders | 强迫及相关障碍 | ocd, obsessive |
| eating_disorders | 喂食与进食障碍 | eating, anorexia |
| personality_disorders | 人格障碍 | personality, borderline |
| neurocognitive_disorders | 神经认知障碍 | dementia, alzheimer |

#### 3.6.2 分类器

使用TransE/RotatE提取的嵌入作为特征，训练分类器：

```python
from sklearn.ensemble import RandomForestClassifier

# 提取疾病节点的嵌入
disease_embeddings = [embeddings[d] for d in labeled_diseases]

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

### 3.7 LLM幻觉检测

#### 3.7.1 任务设计

幻觉检测的核心是比较LLM对真实三元组和伪造三元组的判断：

**真实三元组**：从PrimeKG中随机抽取
**伪造三元组**：通过以下策略生成
1. 替换尾实体：保留(h, r)，替换t
2. 替换头实体：保留(r, t)，替换h
3. 替换关系：保留(h, t)，替换r

#### 3.7.2 自然语言问题生成

将三元组转化为判断题：

```python
templates = [
    "{head} {phrase} {tail}。这个说法是正确的吗？",
    "根据医学知识，{head} {phrase} {tail}。请判断这是否正确。",
    "有一种说法认为：{head} {phrase} {tail}。这是真的吗？"
]
```

#### 3.7.3 评估指标

| 指标 | 说明 |
|------|------|
| 准确率 | 正确判断的样本占总样本的比例 |
| 遗漏率 | 真实三元组被判断为假的比例（假阴性） |
| 检测率 | 伪造三元组被正确识别的比例 |
| 幻觉率 | 1 - 准确率 |

### 3.8 RAG增强问答

#### 3.8.1 RAG架构

```
用户问题 → 向量检索 → Top-K相关块 → LLM生成 → 带证据的回答
```

#### 3.8.2 文本分块

将知识图谱中的信息切分为小块：

```python
chunks = []
for disease, group in disease_groups:
    # 疾病基本信息
    chunks.append(f"{disease}的症状: {', '.join(symptoms)}")
    # 治疗信息
    chunks.append(f"{disease}的药物: {', '.join(drugs)}")
```

#### 3.8.3 向量检索

支持两种向量存储：
1. **ChromaDB**：生产级向量数据库
2. **TF-IDF Fallback**：简单的内存实现

#### 3.8.4 提示词构建

```python
prompt = f"""**问题：**{question}

**Context（来自 PrimeKG 知识图谱）：**
{retrieved_context}

请基于以上Context回答问题。如果Context中没有相关信息，请说明。
"""
```

---

## 四、Web应用设计

### 4.1 四个页面

#### 页面1：精神障碍知识图谱浏览页 (`/graph`)

**功能**：
- 搜索疾病名称
- 展示相关症状、药物、疾病类别
- 关系网络图可视化（Cytoscape.js）

**技术实现**：
- Flask API: `/api/search_disease`
- 前端：Cytoscape.js绘制力导向图

#### 页面2：疾病知识问答页 (`/qa`)

**功能**：
- 自然语言问题输入
- RAG增强回答
- PrimeKG三元组证据展示

**技术实现**：
- Flask API: `/api/answer_question`
- RAG链：检索 + 生成

#### 页面3：LLM幻觉检测页 (`/hallucination`)

**功能**：
- 选择关系类型
- 运行幻觉测试
- 展示判断结果和统计

**技术实现**：
- Flask API: `/api/hallucination_test`
- Chart.js展示统计图表

#### 页面4：模型实验结果页 (`/experiments`)

**功能**：
- TransE vs RotatE性能对比
- MRR、Hits@K雷达图
- 疾病分类准确率

**技术实现**：
- Chart.js绘制对比图表

### 4.2 技术选型

| 组件 | 技术 | 理由 |
|------|------|------|
| Web框架 | Flask | 轻量、Python原生、易于集成 |
| 图可视化 | Cytoscape.js | 功能强大、支持多种布局 |
| 图表 | Chart.js | 轻量、易用、美观 |
| 前端 | Vanilla JS | 无需构建工具、快速原型 |

---

## 五、使用说明

### 5.1 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install pandas numpy networkx pykeen torch scikit-learn flask

# 下载PrimeKG数据
python main.py --step 1
```

### 5.2 运行流程

```bash
# 方式1：运行所有步骤
python main.py --all

# 方式2：分步运行
python main.py --step 2  # 提取子图
python main.py --step 3  # 图分析
python main.py --step 4  # 训练模型
python main.py --step 7  # 幻觉检测
python main.py --step 8  # 构建RAG

# 启动Web应用
python main.py --web
```

### 5.3 访问应用

打开浏览器访问：`http://localhost:5000`

---

## 六、心理学知识映射

### 6.1 DSM-5精神障碍分类

本项目覆盖的主要精神障碍类别：

1. **抑郁障碍** (Depressive Disorders)
   - 重性抑郁障碍 (MDD)
   - 持续性抑郁障碍（恶劣心境）

2. **焦虑障碍** (Anxiety Disorders)
   - 广泛性焦虑障碍 (GAD)
   - 惊恐障碍
   - 社交焦虑障碍
   - 特定恐惧症

3. **创伤及应激相关障碍** (Trauma- and Stressor-Related Disorders)
   - 创伤后应激障碍 (PTSD)
   - 急性应激障碍
   - 适应障碍

4. **精神分裂症谱系及其他精神病性障碍**
   - 精神分裂症
   - 妄想障碍
   - 精神分裂样障碍

5. **双相及相关障碍** (Bipolar and Related Disorders)
   - 双相I型障碍
   - 双相II型障碍

6. **神经发育障碍** (Neurodevelopmental Disorders)
   - 自闭症谱系障碍
   - ADHD

7. **强迫及相关障碍** (Obsessive-Compulsive and Related Disorders)
   - 强迫症 (OCD)
   - 囤积障碍

8. **喂食与进食障碍** (Feeding and Eating Disorders)
   - 神经性厌食症
   - 神经性贪食症
   - 暴食障碍

9. **人格障碍** (Personality Disorders)
   - 边缘型人格障碍
   - 偏执型人格障碍

10. **神经认知障碍** (Neurocognitive Disorders)
    - 阿尔茨海默病
    - 血管性痴呆
    - 路易体痴呆

### 6.2 常见症状节点

| 症状类别 | 示例症状 |
|---------|---------|
| 情绪症状 | 情绪低落、焦虑、情绪波动、易激惹 |
| 认知症状 | 注意力不集中、记忆力下降、思维紊乱 |
| 行为症状 | 社交退缩、活动减少、刻板行为 |
| 躯体症状 | 睡眠障碍、食欲改变、疼痛乏力 |

### 6.3 常用治疗药物

| 药物类别 | 代表药物 |
|---------|---------|
| SSRIs | 氟西汀、舍曲林、帕罗西汀、西酞普兰 |
| SNRIs | 文拉法辛、度洛西汀、去甲文拉法辛 |
| TCAs | 丙米嗪、阿米替林、氯丙米嗪 |
| 非典型抗精神病药 | 奥氮平、利培酮、喹硫平、阿立哌唑 |
| 情感稳定剂 | 锂盐、丙戊酸盐、卡马西平、拉莫三嗪 |
| 苯二氮卓类 | 劳拉西泮、阿普唑仑、氯硝西泮 |

---

## 七、项目定位与免责声明

### 7.1 项目定位

- **教育目的**：心理学课程的精神障碍知识学习工具
- **教学展示**：展示知识图谱、图嵌入、LLM幻觉检测原理
- **科研原型**：知识图谱分析的交互式演示

### 7.2 免责声明

⚠️ **重要提示**：

1. 本系统产生的所有信息仅供参考学习
2. 不用于任何临床诊断或治疗决策
3. 如有心理健康问题，请咨询专业医疗人员
4. 系统回答可能包含不准确信息，用户需自行判断

---

## 八、总结与展望

### 8.1 已完成功能

- ✅ 精神障碍子图提取与分析
- ✅ TransE/RotatE嵌入模型训练与评估
- ✅ 疾病大类预测
- ✅ LLM幻觉检测
- ✅ RAG增强问答
- ✅ 四页Web应用原型

### 8.2 可扩展方向

1. **接入真实LLM API**：使用OpenAI GPT-4进行幻觉检测和问答
2. **更大规模知识图谱**：扩展到其他精神障碍亚型
3. **更丰富的可视化**：添加时序分析、多跳推理可视化
4. **用户交互功能**：添加收藏、反馈、导出等功能

---

*本报告由AI辅助生成，如有问题请联系项目维护者。*
