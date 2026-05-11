# MHKG Analysis System

基于 PrimeKG 的精神健康知识图谱分析与问答系统。

这个仓库汇总了我本次项目的主要成果，包括：
- 精神健康相关子图提取与图统计分析
- 知识图谱嵌入模型训练与对比（TransE / RotatE）
- 疾病类别预测实验
- LLM 幻觉检测实验
- RAG 问答原型
- Flask Web 可视化展示页面

## 项目目标

本项目以 `PrimeKG (Precision Medicine Knowledge Graph)` 为基础，围绕精神健康领域构建一个可分析、可展示、可扩展的知识图谱系统，目标包括：
- 从大型医学知识图谱中抽取精神健康相关子图
- 分析疾病、症状、药物等节点之间的关系结构
- 使用知识图谱嵌入方法学习实体与关系表示
- 评估模型在链接预测和分类任务上的效果
- 设计 LLM 幻觉检测与 RAG 问答原型
- 提供一个可直接运行的 Web 界面用于展示成果

## 仓库结构

```text
.
├─ mental_health_kg/   # 主项目代码
├─ PrimeKG-main/       # PrimeKG 参考仓库/原始资料
├─ data/               # 本地数据目录（默认不上传）
├─ models/             # 本地模型输出目录（默认不上传）
├─ results/            # 本地结果目录（默认不上传）
└─ README.md           # 仓库首页说明
```

`mental_health_kg/` 是本次作业成果的核心目录，主要包含：

```text
mental_health_kg/
├─ analysis/           # 子图提取、图统计分析
├─ models/             # TransE / RotatE / 分类实验
├─ hallucination/      # 幻觉检测数据构造与评估
├─ rag/                # RAG 检索与问答链
├─ web/                # Flask Web 应用
├─ main.py             # 主入口脚本
├─ README.md
├─ PROJECT_REPORT.md
└─ TECHNICAL_DOCUMENTATION.md
```

## 已完成成果

### 1. 精神健康子图构建
- 从 PrimeKG 中筛选抑郁、焦虑、精神分裂症、双相障碍、PTSD、OCD、ADHD、自闭症等相关疾病
- 提取与疾病相关的症状、药物、表型等邻接信息
- 输出精神健康子图 CSV 供后续分析与建模使用

### 2. 图结构分析
- 节点类型分布分析
- 关系类型分布分析
- 度分布可视化
- 中心性分析
- 共病/疾病关联分析

相关结果会输出到本地 `results/` 目录。

### 3. 知识图谱嵌入实验
- 实现并训练 `TransE`
- 实现并训练 `RotatE`
- 使用 PyKEEN 完成训练与评估流程
- 输出 MRR、Hits@K 等指标

### 4. 疾病分类实验
- 基于图嵌入特征进行疾病大类预测
- 支持 Random Forest、Gradient Boosting、SVM 等分类器
- 比较不同分类器在精神疾病类别预测上的表现

### 5. LLM 幻觉检测原型
- 从知识图谱中抽取真实三元组
- 构造伪造三元组作为负样本
- 评估模型对真假医学关系的识别能力

### 6. RAG 问答原型
- 将知识图谱信息转换为可检索文本块
- 构建向量检索/简易检索逻辑
- 为精神健康问答提供知识增强上下文

### 7. Web 展示系统
- 图谱浏览页
- 问答页
- 幻觉检测页
- 实验结果页

当前 Web 系统使用 Flask 实现，适合课程展示和功能演示。

## 技术栈

- Python
- pandas / numpy
- networkx
- PyKEEN
- torch
- scikit-learn
- Flask
- HTML / CSS / JavaScript
- Cytoscape.js
- Chart.js

## 运行方式

建议进入主项目目录运行：

```bash
cd mental_health_kg
```

安装依赖：

```bash
pip install -r requirements.txt
```

按步骤运行：

```bash
python main.py --step 1
python main.py --step 2
python main.py --step 3
python main.py --step 4
```

启动 Web 应用：

```bash
python main.py --web
```

默认访问地址：

```text
http://localhost:5000
```

## 文档说明

如果想看更详细的项目内容，可以继续阅读：
- [mental_health_kg/README.md](mental_health_kg/README.md)
- [mental_health_kg/PROJECT_REPORT.md](mental_health_kg/PROJECT_REPORT.md)
- [mental_health_kg/TECHNICAL_DOCUMENTATION.md](mental_health_kg/TECHNICAL_DOCUMENTATION.md)

## 说明

- `data/`、`models/`、`results/` 默认不提交到仓库，主要避免上传大文件和本地产物
- 本项目定位为课程作业、知识图谱分析原型与展示系统
- 不用于真实临床诊断或医疗决策
