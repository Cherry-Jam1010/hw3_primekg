# Mental Health Knowledge Graph Analysis System
# 精神障碍知识图谱分析与可信问答系统

## 项目简介

本项目基于 **PrimeKG (Precision Medicine Knowledge Graph)** 构建一个面向心理学与精神健康领域的知识图谱分析系统，集成了知识图谱嵌入、LLM幻觉检测、RAG增强问答等功能，并提供友好的Web交互界面。

## 目录结构

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
│   └── rag_chain.py              # RAG 问答链
├── web/                           # Web 应用
│   ├── app.py                     # Flask 主程序
│   ├── templates/                 # HTML 模板
│   └── static/                   # 静态资源
├── results/                       # 结果输出
├── data/                         # 数据目录
├── main.py                       # 主入口
└── README.md                     # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy networkx pykeen torch scikit-learn flask
```

### 2. 下载 PrimeKG 数据

```bash
python main.py --step 1
```

或手动下载：`https://dataverse.harvard.edu/api/access/datafile/6180620`

### 3. 运行完整流程

```bash
python main.py --all
```

### 4. 启动 Web 应用

```bash
python main.py --web
```

访问地址：`http://localhost:5000`

## 功能模块

### 1. 知识图谱子图提取与分析

- 从 PrimeKG 提取精神障碍相关子图
- 分析节点类型分布、关系类型分布
- 识别核心疾病节点（度中心性、PageRank）
- 共病模式分析

### 2. 知识图谱嵌入 (TransE & RotatE)

**TransE**：将关系建模为头实体到尾实体的平移向量
```
h + r ≈ t
```

**RotatE**：将关系建模为复数空间中的旋转
```
h ∘ r = t
```

评估指标：MRR, Hits@1, Hits@3, Hits@5, Hits@10

### 3. 疾病大类预测

基于图嵌入特征，使用分类器预测疾病所属类别（抑郁障碍、焦虑障碍、精神分裂症等）。

### 4. LLM 幻觉检测

- 从知识图谱抽取真实三元组
- 伪造不存在的关系三元组
- 让 LLM 判断真假，评估幻觉率

### 5. RAG 增强问答

- 将知识图谱文本切块存入向量数据库
- 检索相关上下文增强 LLM 回答
- 对比纯 LLM 与 RAG 的回答质量

## Web 应用

四个页面：

1. **精神障碍知识图谱浏览页**：搜索疾病，展示症状、药物、关系网络图
2. **疾病知识问答页**：自然语言提问，RAG 增强回答 + 三元组证据
3. **LLM 幻觉检测页**：展示真假三元组测试结果
4. **模型实验结果页**：TransE vs RotatE 性能对比

## 技术栈

- **数据处理**：pandas, numpy, networkx
- **知识图谱嵌入**：PyKEEN, torch
- **机器学习**：scikit-learn
- **向量数据库**：ChromaDB / 简单 TF-IDF fallback
- **Web 框架**：Flask
- **前端**：HTML5, CSS3, JavaScript, Cytoscape.js, Chart.js

## 心理学知识覆盖

### 精神障碍分类 (DSM-5/ICD-11)

| 大类 | 代表疾病 | 关键词 |
|------|---------|--------|
| 抑郁障碍 | 抑郁症、持续性抑郁障碍 | depression, depressive |
| 焦虑障碍 | 焦虑症、惊恐障碍、社交焦虑 | anxiety, panic, phobia |
| 创伤相关障碍 | PTSD、急性应激障碍 | ptsd, trauma, stress |
| 精神病性障碍 | 精神分裂症、妄想障碍 | schizophrenia, psychosis |
| 双相及相关障碍 | 双相I型、双相II型 | bipolar, manic |
| 神经发育障碍 | 自闭症、ADHD | autism, adhd |
| 强迫及相关障碍 | OCD、囤积障碍 | ocd, obsessive |
| 饮食障碍 | 神经性厌食、暴食症 | eating, anorexia |
| 人格障碍 | 边缘型、偏执型 | personality, borderline |
| 神经认知障碍 | 阿尔茨海默病、痴呆 | dementia, alzheimer |

## 注意事项

⚠️ **免责声明**：本系统仅供心理学课程学习、教学展示和科研辅助使用，不用于真实临床诊断或治疗决策。如有心理健康问题，请咨询专业医疗人员。

## License

MIT License
