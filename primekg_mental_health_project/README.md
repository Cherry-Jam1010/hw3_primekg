# PrimeKG Mental Health Project

基于 `PrimeKG` 重新搭建的课程项目工作区。这个目录和旧的 `mental_health_kg/` 分开维护，当前重点是把 `plan.md` 里的流程拆成可重复执行的阶段。

## Current Status

已经完成的阶段：

1. Step 1 种子疾病发现与精修
2. Step 1 精神健康子图抽取
3. Step 2 三元组导出与训练集划分
4. Step 2 `TransE / RotatE` 基线训练与结果对比
5. Step 3 疾病类别预测基线实验
6. Step 4 幻觉检测数据集与模拟评测基线
7. Step 5 RAG 对比实验基线
8. Step 6 离线 demo 封装

## Structure

```text
primekg_mental_health_project/
├─ data/
│  ├─ processed/
│  │  └─ step2/
│  └─ reports/
├─ results/
├─ src/
│  └─ primekg_mental_health/
│     ├─ __init__.py
│     ├─ config.py
│     └─ pipeline.py
├─ main.py
└─ requirements.txt
```

## Data Source

默认直接复用根目录已有的 PrimeKG 原始数据：

- `../data/primekg_raw.csv`

这样可以避免重复存放接近 1GB 的原始文件。

## Run

执行 Step 1：

```bash
python primekg_mental_health_project/main.py step1
```

执行 Step 2 数据准备：

```bash
python primekg_mental_health_project/main.py step2_prep
```

执行 Step 2 训练：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step2_train --epochs 30 --embedding-dim 64 --batch-size 256 --learning-rate 0.001
```

执行 Step 3 分类实验：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step3_classify --min-class-size 3
```

执行 Step 4 幻觉数据生成：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step4_hallucination --real-per-group 10 --false-per-type 10
```

执行 Step 5 RAG 对比：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step5_rag --top-k 5
```

执行 Step 6 demo 构建：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step6_demo
```

如需在本地启动 demo：

```bash
.venv\Scripts\python.exe primekg_mental_health_project/main.py step6_demo --serve-demo --demo-port 8765
```

如需显式指定 PrimeKG 路径：

```bash
python primekg_mental_health_project/main.py step1 --primekg E:\Grade_2_2\ai\hw3\data\primekg_raw.csv
```

## Step 1 Outputs

- `data/processed/mental_health_seed_candidates.csv`
- `data/processed/mental_health_seed_diseases.csv`
- `data/processed/mental_health_subgraph.csv`
- `data/reports/step1_summary.json`
- `data/reports/step1_summary.md`

说明：

- `mental_health_seed_candidates.csv` 包含全部候选疾病、自动匹配到的家族标签、排除原因和 `keep` 标记
- `mental_health_seed_diseases.csv` 是精修后真正进入实验的疾病种子集
- `mental_health_subgraph.csv` 是后续嵌入、分类、RAG 共用的当前版本子图

## Step 2 Outputs

- `data/processed/step2/triples.tsv`
- `data/processed/step2/train.tsv`
- `data/processed/step2/valid.tsv`
- `data/processed/step2/test.tsv`
- `data/processed/step2/triples_with_labels.csv`
- `data/processed/step2/entities.csv`
- `data/processed/step2/relations.csv`
- `data/processed/step2/disease_labels.csv`
- `data/reports/step2_dataset_summary.json`
- `data/reports/step2_model_comparison.csv`
- `results/kge/transe/metrics.json`
- `results/kge/rotate/metrics.json`
- `data/reports/step3_classification_summary.csv`
- `data/reports/step3_rotate_classification_report.csv`
- `data/reports/step3_transe_classification_report.csv`
- `results/classification/best_model_confusion_matrix.png`
- `data/processed/hallucination/step4_hallucination_dataset.csv`
- `data/reports/step4_dataset_summary.json`
- `data/reports/step4_simulated_llm_predictions.csv`
- `data/reports/step4_simulated_llm_summary.json`
- `results/hallucination/step4_simulated_llm_accuracy.png`
- `data/processed/rag/step5_rag_corpus.csv`
- `data/reports/step5_rag_predictions.csv`
- `data/reports/step5_rag_vs_pure_summary.csv`
- `data/reports/step5_rag_detailed_summary.json`
- `results/rag/step5_rag_vs_pure_accuracy.png`

## Current Dataset Snapshot

当前 Step 1 / Step 2 的实际结果：

- Seed diseases: `76`
- Subgraph edges: `2646`
- Subgraph nodes: `732`
- Triples for training: `2646`
- Entities: `732`
- Relations: `6`
- Train / Valid / Test: `2151 / 248 / 247`

## Current KGE Baseline

基线训练配置：

- Model: `TransE`, `RotatE`
- Epochs: `30`
- Embedding dim: `64`
- Batch size: `256`
- Learning rate: `0.001`
- Device: `cpu`

当前结果：

- `RotatE`: `MRR=0.1632`, `Hits@1=0.0911`, `Hits@3=0.1660`, `Hits@10=0.3198`
- `TransE`: `MRR=0.0158`, `Hits@1=0.0000`, `Hits@3=0.0081`, `Hits@10=0.0344`

当前这版子图上，`RotatE` 明显优于 `TransE`。

## Current Step 3 Baseline

分类设置：

- Embedding source: `RotatE` and `TransE`
- Classifier: `LogisticRegression`
- Validation: `3-fold stratified CV`
- Minimum class size: `3`

分类使用的有效样本：

- Total labeled diseases: `76`
- Missing trained-entity labels: `10`
- Excluded small-class samples: `5`
- Final classification samples: `61`
- Final classes: `7`

类别过滤后保留：

- `anxiety_disorders`
- `autism_adhd_disorders`
- `depressive_disorders`
- `eating_disorders`
- `neurocognitive_disorders`
- `personality_disorders`
- `psychotic_disorders`

当前结果：

- `RotatE`: `accuracy=0.4262`, `macro-F1=0.2889`, `weighted-F1=0.3898`
- `TransE`: `accuracy=0.4262`, `macro-F1=0.2430`, `weighted-F1=0.3923`

这说明在疾病家族分类任务上，`RotatE` 的宏平均表现仍然优于 `TransE`，但当前小样本设置下两者整体准确率相同。

## Current Step 4 Baseline

当前生成内容：

- Real claims: `30`
- False claims: `30`
- Total claims: `60`

假声明类型：

- `polarity_flip`: `10`
- `hierarchy_error`: `10`
- `fabricated`: `10`

关系分组分布：

- `disease_drug`: `23`
- `disease_disease`: `20`
- `disease_phenotype`: `17`

当前还没有接入真实 LLM API，所以这一步先产出了：

- 一套可直接评测的口语化声明数据集
- 一份明确标注为 simulated baseline 的预测结果

当前模拟基线结果：

- Overall accuracy: `0.8000`
- False-claim detection accuracy: `0.7333`
- Real-claim support accuracy: `0.8667`

按假声明类型的模拟结果：

- `fabricated`: `0.90`
- `polarity_flip`: `0.70`
- `hierarchy_error`: `0.60`

这个模式符合预期：完全编造的错误最容易识别，层级类和语义翻转类错误更容易造成“幻觉式接受”。

## Current Step 5 Baseline

当前 RAG 方案：

- Retriever: `TF-IDF`
- Evidence corpus: current PrimeKG subgraph triplets + disease summary docs
- Evaluated on: the `60` Step 4 claims
- RAG verifier rule: retrieval + exact triplet evidence + entity-aware fallback

当前对照结果：

- `rag_verifier`: `accuracy=1.0000`
- `pure_llm_simulated`: `accuracy=0.8000`

当前 RAG 结果细分：

- Real claims: `1.0000`
- False claims: `1.0000`
- `fabricated`: `1.0000`
- `polarity_flip`: `1.0000`
- `hierarchy_error`: `1.0000`

解释：

- 这一步的 RAG 不是生成式回答模型，而是“TF-IDF 检索 + PrimeKG 精确证据校验”的 verifier baseline
- 因此它更接近一个事实核查器，而不是通用聊天式 RAG
- 这个结果适合支撑“结构化证据能显著压低幻觉”的实验结论，但不应表述成“通用 LLM RAG 已完美解决问题”

## Current Step 6 Demo

当前已经新增一个离线本地演示页，用来把 Steps 1-5 的结果整合成更适合汇报和展示的界面。

当前 demo 特点：

- 单文件本地 HTML 页面，可直接打开
- 统一展示 Step 1-5 的核心指标卡片
- 内置 `12` 条经过挑选的 benchmark claim 样例
- 支持按 verdict、relation group 和关键词筛选样例
- 展示检索证据、图表产物和当前 caveats

关键产物：

- `primekg_mental_health_project/demo/index.html`
- `primekg_mental_health_project/demo/demo_payload.json`
- `primekg_mental_health_project/data/reports/step6_demo_summary.json`

说明：

- 这个 demo 目前是 presentation-ready 的离线展示页，不依赖新 Web 框架
- 它重点突出“结构化图谱证据如何帮助事实核查”，而不是做成开放式聊天机器人
- 如需现场展示，推荐直接打开 `demo/index.html`，或者用 `--serve-demo` 启动本地服务

## Dependency Note

当前环境可直接运行：

- `pandas`
- `networkx`
- `scikit-learn`
- `torch`
- `pykeen` in `.venv`
