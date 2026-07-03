from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

RELATION_GROUP_LABELS = {
    "disease_drug": "疾病-药物",
    "disease_phenotype": "疾病-表型",
    "disease_disease": "疾病-疾病",
}

FALSE_TYPE_LABELS = {
    "real": "真实声明",
    "fabricated": "凭空编造",
    "polarity_flip": "语义反转",
    "hierarchy_error": "层级错置",
}

VERDICT_LABELS = {
    "supported": "支持",
    "unsupported": "不支持",
    "unknown": "未知",
}

DISPLAY_RELATION_LABELS = {
    "indication": "适应症",
    "contraindication": "禁忌症",
    "off-label use": "超说明书用药",
    "phenotype present": "表型存在",
    "phenotype absent": "表型缺失",
    "associated with": "相关",
    "parent-child": "上下位",
    "linked to": "关联",
}

NODE_TYPE_LABELS = {
    "disease": "疾病",
    "drug": "药物",
    "effect/phenotype": "表型",
}

FAMILY_LABELS = {
    "anxiety_disorders": "焦虑障碍",
    "autism_adhd_disorders": "自闭症/ADHD",
    "depressive_disorders": "抑郁障碍",
    "eating_disorders": "进食障碍",
    "neurocognitive_disorders": "神经认知障碍",
    "obsessive_compulsive_disorders": "强迫相关障碍",
    "personality_disorders": "人格障碍",
    "psychotic_disorders": "精神病性障碍",
    "trauma_stress_disorders": "创伤应激障碍",
    "bipolar_disorders": "双相障碍",
}

NOTE_TRANSLATIONS = {
    "Exact triplet evidence retrieved.": "已命中精确三元组证据。",
    "Exact triplet exists in corpus and is recovered by entity-aware fallback.": "语料中存在精确三元组，已通过实体感知回退命中。",
    "No exact triplet exists in current PrimeKG subgraph.": "当前 PrimeKG 子图中不存在该精确三元组。",
    "Retriever confidence is too low.": "检索置信度过低。",
    "Relevant evidence not retrieved in top-k.": "Top-k 检索未命中相关证据。",
}


def translate_note(note: str) -> str:
    return NOTE_TRANSLATIONS.get(note, note)


def translate_relation_label(relation: str) -> str:
    return DISPLAY_RELATION_LABELS.get(relation, relation)


def translate_evidence_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    replacements = [
        (" can be used for ", " 可用于 "),
        (" is contraindicated for ", " 对 "),
        (" may be used off-label for ", " 可用于 "),
        (" commonly presents with ", " 常见表现为 "),
        (" typically does not present with ", " 通常不表现为 "),
        (" is a more specific subtype under ", " 是下位类型，隶属于 "),
    ]

    if text.startswith("Disease summary for "):
        disease_name = text.replace("Disease summary for ", "").rstrip(".")
        return f"{disease_name} 的疾病摘要。"

    for english, chinese in replacements:
        if english in text:
            left, right = text.split(english, 1)
            right = right.rstrip(".")
            if english == " is contraindicated for ":
                return f"{left} 对 {right} 属于禁忌。"
            if english == " may be used off-label for ":
                return f"{left} 可用于 {right} 的超说明书场景。"
            return f"{left}{chinese}{right}。"

    if text.startswith("Common phenotypes include "):
        return text.replace("Common phenotypes include ", "常见表型包括 ").rstrip(".") + "。"
    if text.startswith("Associated indication drugs include "):
        return text.replace("Associated indication drugs include ", "相关适应症药物包括 ").rstrip(".") + "。"
    if text.startswith("This disease appears under broader categories such as "):
        return text.replace(
            "This disease appears under broader categories such as ",
            "该疾病在更高层类别中可归于 ",
        ).rstrip(".") + "。"
    return text


def ensure_demo_directories(project_root: Path) -> dict[str, Path]:
    demo_root = project_root / "demo"
    assets_root = demo_root / "assets"
    reports_root = project_root / "data" / "reports"
    demo_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    return {
        "demo": demo_root,
        "assets": assets_root,
        "reports": reports_root,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required demo input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required demo input is missing: {path}")
    return pd.read_csv(path).fillna("")


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def slugify(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def pick_claims(rag_predictions: pd.DataFrame, limit: int) -> list[dict]:
    selected_ids: set[str] = set()
    selected_claim_texts: set[str] = set()
    picked_rows: list[pd.Series] = []

    def add_first_rows(frame: pd.DataFrame, count: int) -> None:
        for _, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            claim_text = str(row["claim_text"]).strip()
            if sample_id in selected_ids:
                continue
            if claim_text in selected_claim_texts:
                continue
            picked_rows.append(row)
            selected_ids.add(sample_id)
            selected_claim_texts.add(claim_text)
            if count <= 1:
                break
            count -= 1

    real_df = rag_predictions.loc[rag_predictions["label"] == "real"].copy()
    false_df = rag_predictions.loc[rag_predictions["label"] == "false"].copy()

    for relation_group in ["disease_drug", "disease_phenotype", "disease_disease"]:
        add_first_rows(
            real_df.loc[real_df["relation_group"] == relation_group].sort_values(
                "top_score", ascending=False
            ),
            2,
        )

    for false_type in ["fabricated", "polarity_flip", "hierarchy_error"]:
        add_first_rows(
            false_df.loc[false_df["false_type"] == false_type].sort_values(
                "top_score", ascending=False
            ),
            2,
        )

    remaining = rag_predictions.sort_values(
        ["is_correct", "top_score"], ascending=[False, False]
    )
    add_first_rows(remaining, max(limit - len(picked_rows), 0))

    display_rows = []
    for row in picked_rows[:limit]:
        evidence = [
            translate_evidence_text(text)
            for text in str(row.get("retrieved_texts", "")).split(" || ")
            if text.strip() and not text.strip().startswith("Disease summary for ")
        ][:4]
        predicted_label = str(row.get("predicted_label", "unknown"))
        target_label = str(row.get("target_label", "unknown"))
        display_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "claim_text": str(row["claim_text"]),
                "relation_group": str(row["relation_group"]),
                "relation_group_label": RELATION_GROUP_LABELS.get(
                    str(row["relation_group"]), str(row["relation_group"])
                ),
                "false_type": str(row["false_type"]),
                "false_type_label": FALSE_TYPE_LABELS.get(
                    str(row["false_type"]), str(row["false_type"])
                ),
                "predicted_label": predicted_label,
                "predicted_label_text": VERDICT_LABELS.get(
                    predicted_label, predicted_label.title()
                ),
                "target_label": target_label,
                "target_label_text": VERDICT_LABELS.get(
                    target_label, target_label.title()
                ),
                "is_correct": str(row.get("is_correct", "")).lower() == "true"
                or bool(row.get("is_correct")),
                "top_score": round(float(row.get("top_score", 0.0)), 3),
                "x_name": str(row["x_name"]),
                "y_name": str(row["y_name"]),
                "display_relation": str(row["display_relation"]),
                "display_relation_label": translate_relation_label(str(row["display_relation"])),
                "note": translate_note(str(row["note"])),
                "evidence": evidence or ["未展示额外证据文本。"],
            }
        )
    return display_rows


def build_scorecards(
    step1_summary: dict,
    kge_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    rag_summary_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    best_kge = kge_df.sort_values("mrr", ascending=False).iloc[0]
    best_cls = classification_df.sort_values("macro_f1", ascending=False).iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]

    headline_metrics = [
        {
            "label": "种子疾病",
            "value": f"{step1_summary['seed_disease_count']}",
            "detail": "筛出的精神健康疾病种子",
            "accent": "teal",
            "progress": min(step1_summary["seed_disease_count"] / 100.0, 1.0),
        },
        {
            "label": "子图边数",
            "value": f"{step1_summary['subgraph_edge_count']:,}",
            "detail": "疾病、药物、表型与层级关系",
            "accent": "amber",
            "progress": min(step1_summary["subgraph_edge_count"] / 3000.0, 1.0),
        },
        {
            "label": "最佳 MRR",
            "value": format_percent(float(best_kge["mrr"])),
            "detail": f"{best_kge['model']} 在当前子图上更优",
            "accent": "coral",
            "progress": min(float(best_kge["mrr"]) / 0.2, 1.0),
        },
        {
            "label": "RAG 准确率",
            "value": format_percent(float(rag_row["accuracy"])),
            "detail": "校验式检索与精确图谱证据",
            "accent": "ink",
            "progress": float(rag_row["accuracy"]),
        },
    ]

    scorecards = [
        {
            "step": "Step 1",
            "title": "Mental Health Subgraph",
            "summary": "PrimeKG was narrowed into a focused subgraph for psychiatric diseases, phenotypes, and medications.",
            "stats": [
                {"label": "Nodes", "value": f"{step1_summary['subgraph_node_count']:,}"},
                {"label": "Edges", "value": f"{step1_summary['subgraph_edge_count']:,}"},
                {"label": "Components", "value": f"{step1_summary['connected_component_count']}"},
            ],
        },
        {
            "step": "Step 2",
            "title": "Embedding Baseline",
            "summary": "RotatE showed a clear advantage over TransE on link prediction for this mental-health slice of PrimeKG.",
            "stats": [
                {"label": "Best Model", "value": str(best_kge["model"])},
                {"label": "MRR", "value": format_percent(float(best_kge["mrr"]))},
                {"label": "Hits@10", "value": format_percent(float(best_kge["hits_at_10"]))},
            ],
        },
        {
            "step": "Step 3",
            "title": "Disease Family Classification",
            "summary": "Disease embeddings retained enough structure to support a multi-class family classification baseline.",
            "stats": [
                {"label": "Best Embedding", "value": str(best_cls["embedding_model"]).upper()},
                {"label": "Accuracy", "value": format_percent(float(best_cls["accuracy"]))},
                {"label": "Macro-F1", "value": format_percent(float(best_cls["macro_f1"]))},
            ],
        },
        {
            "step": "Step 4",
            "title": "Hallucination Benchmark",
            "summary": "The benchmark mixes real claims with fabricated, polarity-flipped, and hierarchy-confused statements.",
            "stats": [
                {"label": "Claims", "value": "60"},
                {"label": "False Types", "value": "3"},
                {"label": "Pure Baseline", "value": format_percent(float(pure_row["accuracy"]))},
            ],
        },
        {
            "step": "Step 5",
            "title": "RAG Verifier",
            "summary": "A lightweight TF-IDF retriever plus exact triplet evidence sharply reduced hallucination compared with the simulated pure baseline.",
            "stats": [
                {"label": "RAG Accuracy", "value": format_percent(float(rag_row["accuracy"]))},
                {
                    "label": "Gain",
                    "value": f"+{(float(rag_row['accuracy']) - float(pure_row['accuracy'])) * 100:.1f} pts",
                },
                {"label": "Hallucination Rate", "value": format_percent(float(rag_row["hallucination_rate"]))},
            ],
        },
        {
            "step": "Step 6",
            "title": "Offline Demo Packaging",
            "summary": "A polished local web demo now ties together metrics, benchmark claims, plots, and evidence inspection in one screen.",
            "stats": [
                {"label": "Mode", "value": "Offline HTML"},
                {"label": "Claims Shown", "value": "12"},
                {"label": "Figures", "value": "3"},
            ],
        },
    ]
    return headline_metrics, scorecards


def build_timeline(
    step1_summary: dict,
    kge_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    rag_summary_df: pd.DataFrame,
) -> list[dict]:
    rotate_row = kge_df.loc[kge_df["model"].str.lower() == "rotate"].iloc[0]
    transe_row = kge_df.loc[kge_df["model"].str.lower() == "transe"].iloc[0]
    rotate_cls = classification_df.loc[
        classification_df["embedding_model"].str.lower() == "rotate"
    ].iloc[0]
    transe_cls = classification_df.loc[
        classification_df["embedding_model"].str.lower() == "transe"
    ].iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]

    return [
        {
            "tag": "Step 1",
            "title": "Refine PrimeKG into a mental-health graph",
            "summary": "Keyword discovery, family rules, and exclusion patterns were combined to isolate a usable psychiatric subgraph.",
            "metric": f"{step1_summary['seed_disease_count']} seed diseases",
        },
        {
            "tag": "Step 2",
            "title": "Train and compare KGE baselines",
            "summary": "TransE and RotatE were trained on 2,646 triples with a coverage-safe train/valid/test split.",
            "metric": f"RotatE MRR {format_percent(float(rotate_row['mrr']))} vs TransE {format_percent(float(transe_row['mrr']))}",
        },
        {
            "tag": "Step 3",
            "title": "Probe embeddings with family classification",
            "summary": "Logistic regression on disease embeddings tested whether model space preserved disease-family structure.",
            "metric": f"Macro-F1: RotatE {format_percent(float(rotate_cls['macro_f1']))}, TransE {format_percent(float(transe_cls['macro_f1']))}",
        },
        {
            "tag": "Step 4",
            "title": "Construct the hallucination benchmark",
            "summary": "The claim set was written in short-video style so the evaluation feels closer to real science-pop content.",
            "metric": "60 claims, 30 real and 30 false",
        },
        {
            "tag": "Step 5",
            "title": "Verify claims with graph-grounded retrieval",
            "summary": "Retrieved evidence is matched against exact graph triples to decide whether a claim is supported or unsupported.",
            "metric": f"Accuracy {format_percent(float(rag_row['accuracy']))} vs simulated pure baseline {format_percent(float(pure_row['accuracy']))}",
        },
        {
            "tag": "Step 6",
            "title": "Package the project into a presentation-ready demo",
            "summary": "This offline page brings the experiment story, benchmark samples, and visual evidence together in one place.",
            "metric": "Local HTML demo with interactive claim explorer",
        },
    ]


def build_family_distribution(step1_summary: dict) -> list[dict]:
    distribution = step1_summary.get("seed_family_distribution", {})
    total = max(sum(int(count) for count in distribution.values()), 1)
    items = []
    for name, count in sorted(distribution.items(), key=lambda item: item[1], reverse=True):
        items.append(
            {
                "name": name,
                "label": FAMILY_LABELS.get(name, name.replace("_", " ")),
                "count": int(count),
                "ratio": round(int(count) / total, 4),
            }
        )
    return items


def build_top_nodes(step1_summary: dict, limit: int = 8) -> list[dict]:
    items = []
    for row in step1_summary.get("top_disease_nodes", [])[:limit]:
        items.append(
            {
                "name": row["node_name"],
                "degree": int(row["degree"]),
            }
        )
    return items


def build_search_suggestions(claims: list[dict]) -> list[dict]:
    suggestion_map: dict[tuple[str, str], dict] = {}

    def add(label: str, value: str, scope: str, hint: str) -> None:
        clean_value = value.strip()
        if not clean_value:
            return
        key = (scope, clean_value.lower())
        if key not in suggestion_map:
            suggestion_map[key] = {
                "label": label,
                "value": clean_value,
                "scope": scope,
                "hint": hint,
            }

    for claim in claims:
        add(claim["x_name"], claim["x_name"], "entity", "实体")
        add(claim["y_name"], claim["y_name"], "entity", "实体")
        add(claim["relation_group_label"], claim["relation_group_label"], "relation", "关系组")
        add(claim["display_relation_label"], claim["display_relation_label"], "relation", "具体关系")
        add(claim["false_type_label"], claim["false_type_label"], "type", "声明类型")
        add(claim["predicted_label_text"], claim["predicted_label_text"], "verdict", "判定结果")

        claim_text = claim["claim_text"].strip()
        if claim_text:
            preview = claim_text if len(claim_text) <= 28 else claim_text[:28] + "..."
            add(preview, claim_text, "claim", "样例声明")

    suggestions = list(suggestion_map.values())
    suggestions.sort(key=lambda item: (item["scope"], len(item["value"]), item["value"].lower()))
    return suggestions


def build_count_distribution(mapping: dict, label_map: dict[str, str] | None = None) -> list[dict]:
    total = max(sum(int(value) for value in mapping.values()), 1)
    rows = []
    for key, value in sorted(mapping.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "key": key,
                "label": label_map.get(key, key) if label_map else key,
                "count": int(value),
                "ratio": round(int(value) / total, 4),
            }
        )
    return rows


def build_model_panels(
    step1_summary: dict,
    step4_dataset_summary: dict,
    step4_simulated_summary: dict,
    rag_summary_df: pd.DataFrame,
    step5_detailed_summary: dict,
    kge_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> dict:
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]
    max_mr = max(float(value) for value in kge_df["mr"].tolist()) if len(kge_df) else 1.0

    return {
        "node_types": build_count_distribution(
            step1_summary.get("node_type_distribution", {}),
            NODE_TYPE_LABELS,
        ),
        "relation_groups": build_count_distribution(
            step1_summary.get("relation_group_distribution", {}),
            RELATION_GROUP_LABELS,
        ),
        "kge_models": [
            {
                "model": row["model"],
                "mrr": float(row["mrr"]),
                "hits_at_10": float(row["hits_at_10"]),
                "mr": float(row["mr"]),
                "mr_score": 1.0 - (float(row["mr"]) / max_mr if max_mr else 0.0),
            }
            for _, row in kge_df.sort_values("mrr", ascending=False).iterrows()
        ],
        "classification_models": [
            {
                "model": str(row["embedding_model"]).upper(),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
            }
            for _, row in classification_df.sort_values("macro_f1", ascending=False).iterrows()
        ],
        "hallucination_counts": [
            {"label": "真实声明", "count": int(step4_dataset_summary["real_samples"])},
            {"label": "错误声明", "count": int(step4_dataset_summary["false_samples"])},
        ],
        "false_type_counts": build_count_distribution(
            {
                key: value
                for key, value in step4_dataset_summary.get("by_false_type", {}).items()
                if key != "real"
            },
            FALSE_TYPE_LABELS,
        ),
        "pure_baseline_breakdown": [
            {
                "label": FALSE_TYPE_LABELS.get(key, key),
                "accuracy": float(value["accuracy"]),
            }
            for key, value in step4_simulated_summary.get("by_false_type", {}).items()
        ],
        "rag_compare": [
            {"label": "纯基线", "accuracy": float(pure_row["accuracy"])},
            {"label": "RAG 校验", "accuracy": float(rag_row["accuracy"])},
        ],
        "rag_breakdown": [
            {
                "label": FALSE_TYPE_LABELS.get(key, key),
                "accuracy": float(value["accuracy"]),
            }
            for key, value in step5_detailed_summary.get("rag_by_false_type", {}).items()
        ],
    }


def build_summary_tables(
    step1_summary: dict,
    step2_dataset_summary: dict,
    kge_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    step4_dataset_summary: dict,
    step4_simulated_summary: dict,
    rag_summary_df: pd.DataFrame,
) -> list[dict]:
    best_kge = kge_df.sort_values("mrr", ascending=False).iloc[0]
    best_cls = classification_df.sort_values("macro_f1", ascending=False).iloc[0]
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]

    return [
        {
            "title": "子图与数据集",
            "rows": [
                {"label": "种子疾病数", "value": str(step1_summary["seed_disease_count"])},
                {"label": "子图节点数", "value": str(step1_summary["subgraph_node_count"])},
                {"label": "子图边数", "value": str(step1_summary["subgraph_edge_count"])},
                {"label": "三元组总数", "value": str(step2_dataset_summary["triple_count"])},
                {"label": "训练/验证/测试", "value": f"{step2_dataset_summary['train_count']} / {step2_dataset_summary['valid_count']} / {step2_dataset_summary['test_count']}"},
            ],
        },
        {
            "title": "模型结果",
            "rows": [
                {"label": "最佳 KGE", "value": str(best_kge["model"])},
                {"label": "最佳 MRR", "value": format_percent(float(best_kge["mrr"]))},
                {"label": "最佳 Hits@10", "value": format_percent(float(best_kge["hits_at_10"]))},
                {"label": "最佳分类模型", "value": str(best_cls["embedding_model"]).upper()},
                {"label": "分类 Macro-F1", "value": format_percent(float(best_cls["macro_f1"]))},
            ],
        },
        {
            "title": "幻觉与校验",
            "rows": [
                {"label": "声明总数", "value": str(step4_dataset_summary["total_samples"])},
                {"label": "真实/错误声明", "value": f"{step4_dataset_summary['real_samples']} / {step4_dataset_summary['false_samples']}"},
                {"label": "纯基线准确率", "value": format_percent(float(pure_row["accuracy"]))},
                {"label": "RAG 校验准确率", "value": format_percent(float(rag_row["accuracy"]))},
                {"label": "层级错置识别率", "value": format_percent(float(step4_simulated_summary["by_false_type"]["hierarchy_error"]["accuracy"]))},
            ],
        },
    ]


def build_process_steps(
    step1_summary: dict,
    step4_dataset_summary: dict,
    rag_summary_df: pd.DataFrame,
    kge_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> list[dict]:
    best_kge = kge_df.sort_values("mrr", ascending=False).iloc[0]
    best_cls = classification_df.sort_values("macro_f1", ascending=False).iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]
    return [
        {"tag": "Step 1", "label": "子图抽取", "metric": f"{step1_summary['seed_disease_count']} 个种子疾病"},
        {"tag": "Step 2", "label": "KGE 训练", "metric": f"{best_kge['model']} MRR {format_percent(float(best_kge['mrr']))}"},
        {"tag": "Step 3", "label": "疾病分类", "metric": f"Macro-F1 {format_percent(float(best_cls['macro_f1']))}"},
        {"tag": "Step 4", "label": "幻觉数据集", "metric": f"{step4_dataset_summary['total_samples']} 条声明"},
        {"tag": "Step 5", "label": "RAG 校验", "metric": f"准确率 {format_percent(float(rag_row['accuracy']))}"},
        {"tag": "Step 6", "label": "演示封装", "metric": "中文可视化看板"},
    ]


def build_gallery(project_root: Path, assets_root: Path) -> list[dict]:
    assets = [
        (
            project_root / "results" / "classification" / "best_model_confusion_matrix.png",
            "classification-confusion.png",
            "最佳分类混淆矩阵",
            "展示疾病家族在当前嵌入空间中的可分性与易混区域。",
        ),
        (
            project_root / "results" / "hallucination" / "step4_simulated_llm_accuracy.png",
            "simulated-baseline.png",
            "Step 4 模拟纯基线",
            "模拟纯基线对凭空编造最敏感，对层级错置最容易出错。",
        ),
        (
            project_root / "results" / "rag" / "step5_rag_vs_pure_accuracy.png",
            "rag-vs-pure.png",
            "RAG 校验 vs 纯基线",
            "图谱证据校验在当前基准集上显著降低幻觉率。",
        ),
    ]
    gallery = []
    for source, target_name, title, caption in assets:
        relative_path = copy_demo_asset(source, assets_root, target_name)
        if relative_path is None:
            continue
        gallery.append(
            {
                "title": title,
                "caption": caption,
                "image": relative_path,
            }
        )
    return gallery


def build_demo_payload(project_root: Path, sample_limit: int = 12) -> dict:
    reports_root = project_root / "data" / "reports"
    step1_summary = load_json(reports_root / "step1_summary.json")
    step2_dataset_summary = load_json(reports_root / "step2_dataset_summary.json")
    kge_df = load_csv(reports_root / "step2_model_comparison.csv")
    classification_df = load_csv(reports_root / "step3_classification_summary.csv")
    step4_dataset_summary = load_json(reports_root / "step4_dataset_summary.json")
    step4_simulated_summary = load_json(reports_root / "step4_simulated_llm_summary.json")
    rag_summary_df = load_csv(reports_root / "step5_rag_vs_pure_summary.csv")
    rag_predictions = load_csv(reports_root / "step5_rag_predictions.csv").sort_values(
        ["label", "false_type", "top_score"], ascending=[True, True, False]
    )

    ensure_demo_directories(project_root)
    headline_metrics, _scorecards = build_scorecards(
        step1_summary=step1_summary,
        kge_df=kge_df,
        classification_df=classification_df,
        rag_summary_df=rag_summary_df,
    )
    claims = pick_claims(rag_predictions, sample_limit)
    payload = {
        "meta": {
            "title": "PrimeKG 精神健康知识图谱实验看板",
            "subtitle": "PrimeKG 驱动的精神健康知识图谱分析与声明校验",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "project_note": "本地离线生成，可直接用于课程展示。",
        },
        "headline_metrics": headline_metrics,
        "summary_tables": build_summary_tables(
            step1_summary=step1_summary,
            step2_dataset_summary=step2_dataset_summary,
            kge_df=kge_df,
            classification_df=classification_df,
            step4_dataset_summary=step4_dataset_summary,
            step4_simulated_summary=step4_simulated_summary,
            rag_summary_df=rag_summary_df,
        ),
        "process_steps": build_process_steps(
            step1_summary,
            step4_dataset_summary,
            rag_summary_df,
            kge_df,
            classification_df,
        ),
        "claims": claims,
        "search_suggestions": build_search_suggestions(claims),
        "family_distribution": build_family_distribution(step1_summary),
        "top_nodes": build_top_nodes(step1_summary),
        "caveats": [
            "Step 4 目前使用的是模拟纯基线，不是实时 LLM API 结果。",
            "Step 5 属于校验式 RAG 基线，核心是 TF-IDF 检索加精确图谱证据，不是开放式生成聊天。",
            "当前子图刻意保持较窄范围，目标是提升可解释性，而不是覆盖全部医学知识。",
        ],
    }
    return payload


def render_demo_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 252, 247, 0.92);
      --line: rgba(24, 37, 42, 0.08);
      --text: #18252a;
      --muted: #5f6d72;
      --teal: #0f766e;
      --amber: #c38718;
      --coral: #c75b39;
      --shadow: 0 18px 56px rgba(26, 44, 49, 0.1);
      --radius-lg: 26px;
      --radius-md: 18px;
      --sans: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
      --serif: "STSong", "SimSun", "Source Han Serif SC", serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(195,135,24,0.16), transparent 28%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }
    .page { max-width: 1260px; margin: 0 auto; padding: 22px 18px 44px; }
    .hero, .section, .claim-list, .detail-panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.65);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .hero {
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 18px;
      padding: 22px;
      border-radius: var(--radius-lg);
    }
    .hero-main {
      position: relative;
      overflow: hidden;
      padding: 22px;
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(247,240,228,0.96));
    }
    .hero-main::after {
      content: "";
      position: absolute;
      right: -30px;
      bottom: -30px;
      width: 180px;
      height: 180px;
      border-radius: 38px;
      transform: rotate(16deg);
      background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(195,135,24,0.12));
    }
    .eyebrow {
      display: inline-flex;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(15,118,110,0.1);
      color: var(--teal);
      font-size: 0.86rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }
    h1 {
      margin: 18px 0 10px;
      font-family: var(--serif);
      font-size: clamp(2.15rem, 4vw, 3.3rem);
      line-height: 1.08;
      letter-spacing: -0.04em;
    }
    .subtitle { margin: 0; color: var(--muted); line-height: 1.7; max-width: 34rem; }
    .hero-strip {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }
    .hero-pill, .metric-card, .panel, .process-card, .claim-card, .detail-box, .caveat, .node-row {
      background: rgba(255,255,255,0.76);
      border: 1px solid var(--line);
    }
    .hero-pill { padding: 14px; border-radius: 16px; }
    .hero-pill span, .metric-card span, .detail-box span { display: block; color: var(--muted); font-size: 0.8rem; margin-bottom: 6px; }
    .hero-pill strong { font-size: 1rem; line-height: 1.45; }
    .hero-side { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-content: start; }
    .metric-card {
      padding: 16px;
      border-radius: 22px;
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 14px;
      align-items: center;
      min-height: 122px;
    }
    .metric-ring {
      --fill: 0.5turn;
      width: 78px;
      height: 78px;
      border-radius: 50%;
      background: conic-gradient(var(--teal) var(--fill), rgba(24,37,42,0.08) 0);
      display: grid;
      place-items: center;
      position: relative;
    }
    .metric-ring::after {
      content: "";
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(24,37,42,0.06);
    }
    .metric-ring[data-accent="coral"] { background: conic-gradient(var(--coral) var(--fill), rgba(24,37,42,0.08) 0); }
    .metric-ring[data-accent="amber"] { background: conic-gradient(var(--amber) var(--fill), rgba(24,37,42,0.08) 0); }
    .metric-ring[data-accent="ink"] { background: conic-gradient(#2f6c7a var(--fill), rgba(24,37,42,0.08) 0); }
    .metric-ring span {
      position: absolute;
      z-index: 1;
      font-size: 0.84rem;
      font-weight: 800;
      color: var(--text);
    }
    .metric-copy strong {
      display: block;
      font-family: var(--serif);
      font-size: 1.6rem;
      line-height: 1;
      margin-bottom: 8px;
    }
    .metric-copy p { margin: 0; color: var(--muted); font-size: 0.9rem; line-height: 1.45; }
    .metric-copy em {
      display: block;
      color: var(--muted);
      font-style: normal;
      font-size: 0.78rem;
      margin-bottom: 6px;
    }
    .section { margin-top: 18px; border-radius: var(--radius-lg); padding: 22px; }
    .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
    .section-head h2 { margin: 0; font-family: var(--serif); font-size: 1.42rem; }
    .section-head span { color: var(--muted); font-size: 0.9rem; }
    .dashboard-grid, .insight-grid, .lab-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .summary-card {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
    }
    .summary-card h3 {
      margin: 0 0 12px;
      font-size: 1rem;
    }
    .summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    .summary-table tr + tr {
      border-top: 1px solid rgba(24,37,42,0.08);
    }
    .summary-table td {
      padding: 10px 0;
      vertical-align: top;
    }
    .summary-table td:first-child {
      color: var(--muted);
      width: 44%;
      padding-right: 12px;
    }
    .summary-table td:last-child {
      color: var(--text);
      font-weight: 700;
      text-align: right;
    }
    .panel { padding: 18px; border-radius: 20px; }
    .panel h3 { margin: 0 0 14px; font-size: 1.05rem; }
    .bar-stack, .claim-list-inner, .node-list, .caveats, .process-row { display: grid; gap: 10px; }
    .bar-row { display: grid; gap: 6px; }
    .bar-label, .tiny-label {
      display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 0.88rem;
    }
    .bar-track { height: 12px; border-radius: 999px; background: rgba(24,37,42,0.08); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--teal), var(--amber)); }
    .bar-fill.alt { background: linear-gradient(90deg, var(--coral), var(--amber)); }
    .process-row { grid-template-columns: repeat(6, minmax(0, 1fr)); }
    .process-card { padding: 14px; border-radius: 18px; position: relative; overflow: hidden; }
    .process-card::after {
      content: "";
      position: absolute;
      top: 50%;
      right: -8px;
      width: 18px;
      height: 18px;
      transform: translateY(-50%) rotate(45deg);
      background: rgba(15,118,110,0.08);
      border-top: 1px solid rgba(24,37,42,0.06);
      border-right: 1px solid rgba(24,37,42,0.06);
    }
    .process-card:last-child::after { display: none; }
    .process-card span { display: block; color: var(--teal); font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; }
    .process-card strong { display: block; font-size: 1rem; margin-bottom: 8px; }
    .process-card p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.45; }
    .lab-toolbar { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; align-items: flex-start; }
    .search-controls { display: grid; grid-template-columns: 160px minmax(260px, 1fr); gap: 10px; flex: 1 1 520px; min-width: 320px; }
    .search-box { position: relative; }
    .search-input {
      flex: 1 1 240px; min-width: 220px; border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; font-size: 0.96rem; background: rgba(255,255,255,0.76); outline: none;
    }
    .search-select {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 0.94rem;
      background: rgba(255,255,255,0.82);
      color: var(--text);
      outline: none;
      appearance: none;
    }
    .suggestion-panel {
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      z-index: 20;
      display: none;
      padding: 10px;
      border-radius: 16px;
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--line);
      box-shadow: 0 16px 36px rgba(24, 37, 42, 0.12);
    }
    .suggestion-panel.is-open { display: block; }
    .suggestion-list { display: grid; gap: 8px; max-height: 280px; overflow: auto; }
    .suggestion-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(244, 239, 230, 0.72);
      color: var(--text);
      text-align: left;
      cursor: pointer;
    }
    .suggestion-item:hover { background: rgba(15, 118, 110, 0.1); }
    .suggestion-label { font-size: 0.92rem; line-height: 1.45; }
    .suggestion-hint { color: var(--muted); font-size: 0.8rem; white-space: nowrap; }
    .filter-group { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      border: 0; border-radius: 999px; padding: 10px 14px; background: rgba(24,37,42,0.08);
      color: var(--text); font-weight: 700; cursor: pointer;
    }
    .chip.is-active { background: var(--text); color: #fff; }
    .claim-list, .detail-panel { border-radius: 22px; padding: 14px; min-height: 540px; }
    .claim-list-inner { max-height: 680px; overflow: auto; padding-right: 4px; }
    .claim-card { padding: 16px; border-radius: 18px; cursor: pointer; }
    .claim-card.is-active { border-color: rgba(15,118,110,0.22); box-shadow: 0 12px 28px rgba(15,118,110,0.08); }
    .claim-meta, .detail-top { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .badge { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
    .badge.supported { background: rgba(15,118,110,0.12); color: var(--teal); }
    .badge.unsupported { background: rgba(199,91,57,0.12); color: var(--coral); }
    .badge.unknown { background: rgba(34,61,71,0.12); color: #223d47; }
    .badge.soft { background: rgba(24,37,42,0.08); color: var(--muted); }
    .claim-card h3, .detail-panel h3 { margin: 0; line-height: 1.55; font-size: 1rem; }
    .claim-card p { margin: 10px 0 0; color: var(--muted); font-size: 0.9rem; }
    .detail-note { margin: 14px 0; padding: 12px 14px; border-radius: 14px; background: rgba(15,118,110,0.08); color: #223d47; line-height: 1.6; }
    .detail-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0 18px; }
    .detail-box { padding: 12px; border-radius: 16px; }
    .detail-box strong { display: block; font-size: 0.96rem; line-height: 1.45; }
    .evidence-list { display: grid; gap: 10px; }
    .evidence-item {
      padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,0.78);
      border-left: 4px solid var(--amber); line-height: 1.6;
    }
    .node-row { display: flex; justify-content: space-between; gap: 10px; padding: 12px 14px; border-radius: 14px; }
    .node-row strong { max-width: 78%; line-height: 1.45; font-size: 0.92rem; }
    .node-row span { white-space: nowrap; color: var(--muted); font-size: 0.86rem; }
    .caveat { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.55; }
    .caveat { padding: 12px 14px; border-radius: 14px; }
    .empty-state { padding: 30px 14px; text-align: center; color: var(--muted); }
    @media (max-width: 1024px) {
      .hero, .dashboard-grid, .insight-grid, .lab-grid, .summary-grid { grid-template-columns: 1fr; }
      .process-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero-side { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 680px) {
      .page { padding: 16px 12px 36px; }
      .hero-strip, .detail-grid, .process-row { grid-template-columns: 1fr; }
      .hero-side { grid-template-columns: 1fr; }
      .metric-card { grid-template-columns: 68px 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <section class="hero-main">
        <div class="eyebrow">PrimeKG 精神健康实验看板</div>
        <h1>知识图谱、嵌入模型与声明校验</h1>
        <p class="subtitle">__SUBTITLE__</p>
        <div class="hero-strip">
          <div class="hero-pill"><span>展示模式</span><strong>中文离线可视化</strong></div>
          <div class="hero-pill"><span>样例声明</span><strong>12 条精选核查样例</strong></div>
          <div class="hero-pill"><span>生成时间</span><strong>__GENERATED_AT__</strong></div>
        </div>
      </section>
      <aside class="hero-side" id="headlineMetrics"></aside>
    </header>

    <section class="section">
      <div class="section-head"><h2>关键指标</h2><span>保留核心结果，改为清晰表格</span></div>
      <div class="summary-grid" id="summaryTables"></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>流程路径</h2><span>对应 plan.md 的 6 个阶段</span></div>
      <div class="process-row" id="processSteps"></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>样例核查台</h2><span>筛选并查看图谱证据</span></div>
      <div class="lab-toolbar">
        <div class="search-controls">
          <select id="searchScope" class="search-select">
            <option value="all">全部范围</option>
            <option value="entity">疾病 / 药物</option>
            <option value="relation">关系类型</option>
            <option value="type">声明类型</option>
            <option value="verdict">判定结果</option>
            <option value="claim">样例声明</option>
          </select>
          <div class="search-box">
            <input id="searchInput" class="search-input" type="text" placeholder="搜索疾病、药物或声明关键词">
            <div id="suggestionPanel" class="suggestion-panel">
              <div id="suggestionList" class="suggestion-list"></div>
            </div>
          </div>
        </div>
        <div class="filter-group" id="verdictFilters"></div>
        <div class="filter-group" id="groupFilters"></div>
      </div>
      <div class="lab-grid">
        <section class="claim-list"><div class="claim-list-inner" id="claimList"></div></section>
        <section class="detail-panel" id="claimDetail"></section>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><h2>图谱侧重点</h2><span>家族分布与高连接疾病节点</span></div>
      <div class="insight-grid">
        <article class="panel"><h3>种子疾病家族分布</h3><div class="bar-stack" id="familyBars"></div></article>
        <article class="panel"><h3>高连接疾病节点</h3><div class="node-list" id="topNodes"></div></article>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><h2>说明边界</h2><span>保留必要但简短的提示</span></div>
      <div class="caveats" id="caveats"></div>
    </section>
  </div>

  <script>
    const payload = __PAYLOAD_JSON__;
    const state = {
      verdict: "all",
      group: "all",
      search: "",
      searchScope: "all",
      activeId: payload.claims.length ? payload.claims[0].sample_id : null
    };
    const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

    function makeChip(label, value, current, onClick) {
      const button = document.createElement("button");
      button.className = "chip" + (value === current ? " is-active" : "");
      button.textContent = label;
      button.addEventListener("click", () => onClick(value));
      return button;
    }

    function renderHeadlineMetrics() {
      const root = document.getElementById("headlineMetrics");
      root.innerHTML = "";
      payload.headline_metrics.forEach((item) => {
        const div = document.createElement("div");
        div.className = "metric-card";
        const progress = Math.max(0.06, Math.min(item.progress || 0, 1));
        div.innerHTML = `
          <div class="metric-ring" data-accent="${item.accent}" style="--fill:${progress}turn">
            <span>${Math.round(progress * 100)}%</span>
          </div>
          <div class="metric-copy">
            <em>${item.label}</em>
            <strong>${item.value}</strong>
            <p>${item.detail}</p>
          </div>
        `;
        root.appendChild(div);
      });
    }

    function renderSummaryTables() {
      const root = document.getElementById("summaryTables");
      root.innerHTML = "";
      payload.summary_tables.forEach((table) => {
        const card = document.createElement("article");
        card.className = "summary-card";
        card.innerHTML = `
          <h3>${table.title}</h3>
          <table class="summary-table">
            <tbody>
              ${table.rows.map((row) => `
                <tr>
                  <td>${row.label}</td>
                  <td>${row.value}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `;
        root.appendChild(card);
      });
    }

    function renderDonut(rootId, items, palette) {
      const root = document.getElementById(rootId);
      root.innerHTML = "";
      const total = items.reduce((sum, item) => sum + Number(item.count || 0), 0);
      let cursor = 0;
      const stops = items.map((item, index) => {
        const ratio = Number(item.ratio || 0);
        const start = cursor * 100;
        cursor += ratio;
        const end = cursor * 100;
        const color = palette[index % palette.length];
        item._color = color;
        return `${color} ${start.toFixed(2)}% ${end.toFixed(2)}%`;
      });
      const wrap = document.createElement("div");
      wrap.className = "donut-wrap";
      wrap.innerHTML = `
        <div class="donut-chart" style="background: conic-gradient(${stops.join(", ")})">
          <div class="donut-center">
            <strong>${total}</strong>
            <span>总量</span>
          </div>
        </div>
        <div class="legend-list">
          ${items.map((item) => `
            <div class="legend-row">
              <span class="legend-dot" style="background:${item._color}"></span>
              <span class="legend-name">${item.label}</span>
              <span class="legend-value">${item.count}</span>
            </div>
          `).join("")}
        </div>
      `;
      root.appendChild(wrap);
    }

    function renderSimpleBars(rootId, items, valueKey, formatter, alt = false) {
      const root = document.getElementById(rootId);
      root.innerHTML = "";
      items.forEach((item) => {
        const value = item[valueKey];
        const width = valueKey === "count" ? Math.max(item.ratio * 100, 6) : Math.max(value * 100, 6);
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-label"><span>${item.label}</span><span>${formatter(value, item)}</span></div>
          <div class="bar-track"><div class="bar-fill ${alt ? "alt" : ""}" style="width:${width}%"></div></div>
        `;
        root.appendChild(row);
      });
    }

    function renderModelCompare(rootId, items, leftLabel, leftKey, rightLabel, rightKey) {
      const root = document.getElementById(rootId);
      root.innerHTML = "";
      items.forEach((item) => {
        const div = document.createElement("div");
        div.className = "dual-bar-card";
        div.innerHTML = `
          <div class="dual-bar-head"><span>${item.model}</span><span>${formatPercent(item[leftKey])}</span></div>
          <div class="tiny-label"><span>${leftLabel}</span><span>${formatPercent(item[leftKey])}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(item[leftKey] * 100, 4)}%"></div></div>
          <div class="tiny-label"><span>${rightLabel}</span><span>${formatPercent(item[rightKey])}</span></div>
          <div class="bar-track"><div class="bar-fill alt" style="width:${Math.max(item[rightKey] * 100, 4)}%"></div></div>
        `;
        root.appendChild(div);
      });
    }

    function renderRadar(rootId, items, metrics, palette) {
      const root = document.getElementById(rootId);
      root.innerHTML = "";

      const canvas = document.createElement("canvas");
      canvas.className = "radar-canvas";
      canvas.width = 180;
      canvas.height = 180;

      const legend = document.createElement("div");
      legend.className = "radar-legend";
      legend.innerHTML = items.map((item, index) => `
        <div class="radar-item">
          <strong style="color:${palette[index % palette.length]}">${item.model}</strong>
          <span>${metrics.map((metric) => `${metric.label} ${formatPercent(item[metric.key])}`).join(" · ")}</span>
        </div>
      `).join("");

      const panel = document.createElement("div");
      panel.className = "radar-panel";
      panel.appendChild(canvas);
      panel.appendChild(legend);
      root.appendChild(panel);

      const ctx = canvas.getContext("2d");
      const cx = 90;
      const cy = 90;
      const radius = 58;
      const levels = 4;
      const angleStep = (Math.PI * 2) / metrics.length;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(24, 37, 42, 0.12)";
      ctx.fillStyle = "rgba(24, 37, 42, 0.42)";
      ctx.lineWidth = 1;

      for (let level = 1; level <= levels; level++) {
        const r = (radius / levels) * level;
        ctx.beginPath();
        metrics.forEach((metric, idx) => {
          const angle = -Math.PI / 2 + angleStep * idx;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
      }

      metrics.forEach((metric, idx) => {
        const angle = -Math.PI / 2 + angleStep * idx;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(24, 37, 42, 0.6)";
        ctx.font = "11px Microsoft YaHei";
        ctx.textAlign = x >= cx ? "left" : "right";
        ctx.fillText(metric.label, cx + Math.cos(angle) * (radius + 14), cy + Math.sin(angle) * (radius + 14));
      });

      items.forEach((item, itemIndex) => {
        ctx.beginPath();
        metrics.forEach((metric, idx) => {
          const angle = -Math.PI / 2 + angleStep * idx;
          const value = Math.max(0, Math.min(item[metric.key], 1));
          const x = cx + Math.cos(angle) * radius * value;
          const y = cy + Math.sin(angle) * radius * value;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = palette[itemIndex % palette.length] + "33";
        ctx.strokeStyle = palette[itemIndex % palette.length];
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
      });
    }

    function renderProcessSteps() {
      const root = document.getElementById("processSteps");
      root.innerHTML = "";
      payload.process_steps.forEach((item) => {
        const div = document.createElement("article");
        div.className = "process-card";
        div.innerHTML = `<span>${item.tag}</span><strong>${item.label}</strong><p>${item.metric}</p>`;
        root.appendChild(div);
      });
    }

    function scopeMatchedSuggestions() {
      const query = state.search.trim().toLowerCase();
      const scoped = payload.search_suggestions.filter((item) => state.searchScope === "all" || item.scope === state.searchScope);
      if (!query) {
        return scoped.slice(0, 8);
      }
      return scoped
        .filter((item) => item.value.toLowerCase().includes(query) || item.label.toLowerCase().includes(query))
        .slice(0, 8);
    }

    function renderSuggestions() {
      const panel = document.getElementById("suggestionPanel");
      const list = document.getElementById("suggestionList");
      const items = scopeMatchedSuggestions();
      list.innerHTML = "";
      if (!items.length) {
        panel.classList.remove("is-open");
        return;
      }
      items.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion-item";
        button.innerHTML = `<span class="suggestion-label">${item.label}</span><span class="suggestion-hint">${item.hint}</span>`;
        button.addEventListener("click", () => {
          state.search = item.value;
          document.getElementById("searchInput").value = item.value;
          panel.classList.remove("is-open");
          renderClaims();
        });
        list.appendChild(button);
      });
      panel.classList.add("is-open");
    }

    function renderFilters() {
      const verdictRoot = document.getElementById("verdictFilters");
      const groupRoot = document.getElementById("groupFilters");
      verdictRoot.innerHTML = "";
      groupRoot.innerHTML = "";
      [["全部声明","all"],["支持","supported"],["不支持","unsupported"]].forEach(([label, value]) => {
        verdictRoot.appendChild(makeChip(label, value, state.verdict, (next) => { state.verdict = next; renderClaims(); }));
      });
      const groups = [["全部关系","all"], ...Array.from(new Set(payload.claims.map((claim) => claim.relation_group))).map((group) => [payload.claims.find((claim) => claim.relation_group === group)?.relation_group_label || group, group])];
      groups.forEach(([label, value]) => {
        groupRoot.appendChild(makeChip(label, value, state.group, (next) => { state.group = next; renderClaims(); }));
      });
    }

    function getFilteredClaims() {
      const search = state.search.trim().toLowerCase();
      return payload.claims.filter((claim) => {
        const verdictPass = state.verdict === "all" || claim.predicted_label === state.verdict;
        const groupPass = state.group === "all" || claim.relation_group === state.group;
        const scopeText = {
          all: [claim.claim_text, claim.x_name, claim.y_name, claim.relation_group_label, claim.display_relation_label, claim.false_type_label, claim.note].join(" "),
          entity: [claim.x_name, claim.y_name].join(" "),
          relation: [claim.relation_group_label, claim.display_relation_label].join(" "),
          type: claim.false_type_label,
          verdict: claim.predicted_label_text,
          claim: claim.claim_text,
        };
        const haystack = (scopeText[state.searchScope] || scopeText.all).toLowerCase();
        const searchPass = !search || haystack.includes(search);
        return verdictPass && groupPass && searchPass;
      });
    }

    function renderClaimDetail(claim) {
      const root = document.getElementById("claimDetail");
      if (!claim) {
        root.innerHTML = `<div class="empty-state">当前筛选条件下没有匹配到声明。</div>`;
        return;
      }
      root.innerHTML = `
        <div class="detail-top">
          <span class="badge ${claim.predicted_label}">${claim.predicted_label_text}</span>
          <span class="badge soft">${claim.relation_group_label}</span>
          <span class="badge soft">${claim.false_type_label}</span>
        </div>
        <h3>${claim.claim_text}</h3>
        <div class="detail-note">${claim.note}</div>
        <div class="detail-grid">
          <div class="detail-box"><span>实体对</span><strong>${claim.x_name} ↔ ${claim.y_name}</strong></div>
          <div class="detail-box"><span>关系</span><strong>${claim.display_relation_label}</strong></div>
          <div class="detail-box"><span>检索分数</span><strong>${claim.top_score}</strong></div>
        </div>
        <h3 style="font-size:1.1rem;margin:0 0 12px;">证据片段</h3>
        <div class="evidence-list">${claim.evidence.map((item) => `<div class="evidence-item">${item}</div>`).join("")}</div>
      `;
    }

    function renderClaims() {
      const claims = getFilteredClaims();
      const list = document.getElementById("claimList");
      list.innerHTML = "";
      if (!claims.length) {
        state.activeId = null;
        list.innerHTML = `<div class="empty-state">没有符合条件的声明。</div>`;
        renderClaimDetail(null);
        return;
      }
      if (!claims.some((claim) => claim.sample_id === state.activeId)) {
        state.activeId = claims[0].sample_id;
      }
      claims.forEach((claim) => {
        const article = document.createElement("article");
        article.className = "claim-card" + (claim.sample_id === state.activeId ? " is-active" : "");
        article.innerHTML = `
          <div class="claim-meta">
            <span class="badge ${claim.predicted_label}">${claim.predicted_label_text}</span>
            <span class="badge soft">${claim.relation_group_label}</span>
            <span class="badge soft">${claim.false_type_label}</span>
          </div>
          <h3>${claim.claim_text}</h3>
          <p>${claim.note}</p>
        `;
        article.addEventListener("click", () => { state.activeId = claim.sample_id; renderClaims(); });
        list.appendChild(article);
      });
      renderClaimDetail(claims.find((claim) => claim.sample_id === state.activeId) || claims[0]);
    }

    function renderTopNodes() {
      const root = document.getElementById("topNodes");
      root.innerHTML = "";
      payload.top_nodes.forEach((item) => {
        const div = document.createElement("div");
        div.className = "node-row";
        div.innerHTML = `<strong>${item.name}</strong><span>度数 ${item.degree}</span>`;
        root.appendChild(div);
      });
    }

    function renderCaveats() {
      const root = document.getElementById("caveats");
      root.innerHTML = "";
      payload.caveats.forEach((text) => {
        const div = document.createElement("div");
        div.className = "caveat";
        div.textContent = text;
        root.appendChild(div);
      });
    }

    document.getElementById("searchInput").addEventListener("input", (event) => {
      state.search = event.target.value;
      renderSuggestions();
      renderClaims();
    });
    document.getElementById("searchInput").addEventListener("focus", () => {
      renderSuggestions();
    });
    document.getElementById("searchScope").addEventListener("change", (event) => {
      state.searchScope = event.target.value;
      renderSuggestions();
      renderClaims();
    });
    document.addEventListener("click", (event) => {
      const panel = document.getElementById("suggestionPanel");
      const box = document.querySelector(".search-box");
      if (!box.contains(event.target)) {
        panel.classList.remove("is-open");
      }
    });

    renderHeadlineMetrics();
    renderSummaryTables();
    renderProcessSteps();
    renderFilters();
    renderClaims();
    renderSimpleBars("familyBars", payload.family_distribution, "count", (value) => value);
    renderTopNodes();
    renderCaveats();
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", payload["meta"]["title"])
        .replace("__SUBTITLE__", payload["meta"]["subtitle"])
        .replace("__GENERATED_AT__", payload["meta"]["generated_at"])
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def write_demo_files(project_root: Path, payload: dict) -> tuple[Path, Path]:
    paths = ensure_demo_directories(project_root)
    demo_root = paths["demo"]
    payload_path = demo_root / "demo_payload.json"
    html_path = demo_root / "index.html"

    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_demo_html(payload), encoding="utf-8")
    return html_path, payload_path


def write_demo_summary(project_root: Path, payload: dict, html_path: Path, payload_path: Path) -> Path:
    summary = {
        "html_path": str(html_path),
        "payload_path": str(payload_path),
        "claim_count": len(payload["claims"]),
        "gallery_count": len(payload["gallery"]),
        "generated_at": payload["meta"]["generated_at"],
        "note": "Open demo/index.html directly or serve the demo folder with a local HTTP server.",
    }
    summary_path = project_root / "data" / "reports" / "step6_demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def serve_demo_directory(demo_root: Path, host: str, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(demo_root))
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Serving demo at http://{host}:{port}/")
    print("Press Ctrl+C to stop the local demo server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def run_demo_pipeline(
    project_root: Path,
    *,
    sample_limit: int = 12,
    serve: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> Path:
    project_root = project_root.resolve()
    payload = build_demo_payload(project_root, sample_limit=sample_limit)
    html_path, payload_path = write_demo_files(project_root, payload)
    summary_path = write_demo_summary(project_root, payload, html_path, payload_path)
    if serve:
        serve_demo_directory(html_path.parent, host=host, port=port)
    return summary_path
