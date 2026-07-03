from __future__ import annotations

import json
import re
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
    "partial": "部分支持",
    "unsupported": "不支持",
    "uncovered": "无相关记录",
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

STYLE_KEY_MAP = {
    "克制版": "restrained",
    "冲击版": "impact",
    "学术版": "academic",
}

STYLE_LABELS = {
    "restrained": "克制版",
    "impact": "冲击版",
    "academic": "学术版",
}

STYLE_THEMES = {
    "restrained": {"accent": "#90f9df", "accent_soft": "rgba(144,249,223,0.18)", "glow": "#d5a13f"},
    "impact": {"accent": "#ff7f50", "accent_soft": "rgba(255,127,80,0.18)", "glow": "#ffca6f"},
    "academic": {"accent": "#8cd4ff", "accent_soft": "rgba(140,212,255,0.18)", "glow": "#a8f0ff"},
}

RATING_ORDER = [
    "真实",
    "轻度夸大",
    "因果颠倒",
    "完全编造",
    "语义鸿沟",
    "否定性问题",
]

NOTE_TRANSLATIONS = {
    "Exact triplet evidence retrieved.": "已命中精确三元组证据。",
    "Exact triplet exists in corpus and is recovered by entity-aware fallback.": "语料中存在精确三元组，已通过实体感知回退命中。",
    "No exact triplet exists in current PrimeKG sub图。": "当前 PrimeKG 子图中不存在该精确三元组。",
    "No exact triplet exists in current PrimeKG subgraph.": "当前 PrimeKG 子图中不存在该精确三元组。",
    "Retriever confidence is too low.": "检索置信度过低。",
    "Relevant evidence not retrieved in top-k.": "Top-k 检索未命中相关证据。",
}

TEXT_REPLACEMENTS = [
    ("inclusion body myopathy with Paget disease of bone and frontotemporal dementia", "包涵体肌病-佩吉特骨病-额颞叶痴呆综合征"),
    ("inclusion body myopathy with early-onset Paget disease with or without frontotemporal dementia", "早发佩吉特骨病相关包涵体肌病（可伴额颞叶痴呆）"),
    ("PRKAR1B-related neurodegenerative dementia with intermediate filaments", "PRKAR1B相关中间丝神经变性性痴呆"),
    ("frontotemporal dementia with motor neuron disease", "伴运动神经元病的额颞叶痴呆"),
    ("behavioral variant of frontotemporal dementia", "行为变异型额颞叶痴呆"),
    ("attention deficit-hyperactivity disorder", "注意缺陷多动障碍（ADHD）"),
    ("treatment-refractory schizophrenia", "难治性精神分裂症"),
    ("early-onset schizophrenia", "早发性精神分裂症"),
    ("parkinsonism with dementia of Guadeloupe", "瓜德罗普帕金森综合征伴痴呆"),
    ("schizoaffective disorder", "分裂情感性障碍"),
    ("atypical depressive disorder", "非典型抑郁障碍"),
    ("major depressive disorder", "重度抑郁障碍"),
    ("semantic dementia", "语义性痴呆"),
    ("frontotemporal dementia", "额颞叶痴呆"),
    ("metabolic disease with dementia", "伴痴呆的代谢性疾病"),
    ("cerebral lipidosis with dementia", "脑脂质沉积症伴痴呆"),
    ("vascular dementia", "血管性痴呆"),
    ("genetic dementia", "遗传性痴呆"),
    ("Alzheimer disease", "阿尔茨海默病"),
    ("paranoid schizophrenia", "偏执型精神分裂症"),
    ("schizophrenia", "精神分裂症"),
    ("anxiety disorder", "焦虑障碍"),
    ("personality disorder", "人格障碍"),
    ("obsessive-compulsive disorder", "强迫症"),
    ("anorexia nervosa", "神经性厌食症"),
    ("endogenous depression", "内源性抑郁"),
    ("Specific learning disability", "特定学习障碍"),
    ("Behavioral abnormality", "行为异常"),
    ("Deposits immunoreactive to beta-amyloid protein", "β-淀粉样蛋白免疫反应沉积"),
    ("Diminished motivation", "动机减退"),
    ("Decreased male libido", "男性性欲减退"),
    ("Decreased female libido", "女性性欲减退"),
    ("Hallucinations", "幻觉"),
    ("Myositis", "肌炎"),
    ("Mania", "躁狂"),
    ("Rigidity", "肌强直"),
    ("Amyotrophic lateral sclerosis", "肌萎缩侧索硬化"),
    ("Brain atrophy", "脑萎缩"),
    ("Attention deficit hyperactivity disorder", "注意缺陷多动障碍"),
    ("attention deficit hyperactivity disorder, inattentive type", "注意缺陷多动障碍·注意缺陷型"),
    ("Frontotemporal dementia", "额颞叶痴呆"),
    ("Dementia", "痴呆"),
    ("Depressivity", "抑郁性情绪"),
    ("Anxiety", "焦虑"),
    ("Alexia", "失读症"),
    ("Ethynodiol diacetate", "炔诺二醇醋酸酯"),
    ("Desvenlafaxine", "去甲文拉法辛"),
    ("Pramiracetam", "普拉西坦"),
    ("Ergotamine", "麦角胺"),
    ("Ketamine", "氯胺酮"),
    ("Butabarbital", "布他巴比妥"),
    ("Setiptiline", "司替普汀"),
    ("Disulfiram", "双硫仑"),
    ("Doxepin", "多塞平"),
    ("Prochlorperazine", "丙氯拉嗪"),
    ("Carisoprodol", "卡立普多"),
    ("Valproic acid", "丙戊酸"),
    ("Fluoxetine", "氟西汀"),
    ("Paroxetine", "帕罗西汀"),
    ("Olanzapine", "奥氮平"),
    ("Phenacetin", "非那西汀"),
]


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


def translate_note(note: str) -> str:
    return NOTE_TRANSLATIONS.get(note, note)


def translate_relation_label(relation: str) -> str:
    return DISPLAY_RELATION_LABELS.get(relation, relation)


def translate_terms(text: str) -> str:
    output = str(text)
    for source, target in TEXT_REPLACEMENTS:
        output = output.replace(source, target)
    return output


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        clean = value.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        output.append(clean)
    return output


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
        return translate_terms(f"{disease_name} 的疾病摘要。")

    for english, chinese in replacements:
        if english not in text:
            continue
        left, right = text.split(english, 1)
        right = right.rstrip(".")
        if english == " is contraindicated for ":
            return translate_terms(f"{left} 对 {right} 属于禁忌。")
        if english == " may be used off-label for ":
            return translate_terms(f"{left} 可用于 {right} 的超说明书场景。")
        return translate_terms(f"{left}{chinese}{right}。")

    if text.startswith("Common phenotypes include "):
        return translate_terms(text.replace("Common phenotypes include ", "常见表型包括 ").rstrip(".") + "。")
    if text.startswith("Associated indication drugs include "):
        return translate_terms(text.replace("Associated indication drugs include ", "相关适应症药物包括 ").rstrip(".") + "。")
    if text.startswith("This disease appears under broader categories such as "):
        return translate_terms(text.replace(
            "This disease appears under broader categories such as ",
            "该疾病在更高层类别中可归于 ",
        ).rstrip(".") + "。")
    return translate_terms(text)


def ensure_demo_directories(project_root: Path) -> dict[str, Path]:
    demo_root = project_root / "demo"
    reports_root = project_root / "data" / "reports"
    demo_root.mkdir(parents=True, exist_ok=True)
    return {"demo": demo_root, "reports": reports_root}


def extract_heading_block(text: str, heading_prefix: str, stop_prefixes: list[str]) -> str:
    lines = text.splitlines()
    start_idx = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith(heading_prefix):
            start_idx = index + 1
            break
    if start_idx is None:
        return ""

    output: list[str] = []
    for raw_line in lines[start_idx:]:
        stripped = raw_line.strip()
        if any(stripped.startswith(prefix) for prefix in stop_prefixes):
            break
        output.append(raw_line)
    return "\n".join(output).strip()


def extract_table_after_marker(text: str, marker: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        if marker not in raw_line:
            continue
        table_lines: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor].strip()
            if not candidate:
                if table_lines:
                    break
                cursor += 1
                continue
            if not candidate.startswith("|"):
                if table_lines:
                    break
                cursor += 1
                continue
            table_lines.append(candidate)
            cursor += 1
        if table_lines:
            return parse_markdown_table("\n".join(table_lines))
    return []


def extract_bullets_after_marker(text: str, marker: str) -> list[str]:
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        if marker not in raw_line:
            continue
        bullets: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor].strip()
            if not candidate:
                if bullets:
                    break
                cursor += 1
                continue
            if candidate.startswith("- "):
                bullets.append(candidate[2:].strip())
                cursor += 1
                continue
            if bullets:
                break
            cursor += 1
        if bullets:
            return bullets
    return []


def parse_markdown_table(block: str) -> list[dict[str, str]]:
    rows = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return []
    headers = [cell.strip() for cell in rows[0].strip("|").split("|")]
    output: list[dict[str, str]] = []
    for line in rows[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        output.append(dict(zip(headers, cells)))
    return output


def parse_version_bullets(section_text: str) -> dict[str, list[str]]:
    versions = {style_key: [] for style_key in STYLE_LABELS}
    pattern = re.compile(r"\*\*版本([ABC])\s*·\s*(克制版|冲击版|学术版)\*\*\s*((?:\n- .+)+)", re.MULTILINE)
    for match in pattern.finditer(section_text):
        style_key = STYLE_KEY_MAP[match.group(2)]
        versions[style_key] = [
            line[2:].strip()
            for line in match.group(3).splitlines()
            if line.strip().startswith("- ")
        ]
    return versions


def parse_version_line_items(lines: list[str]) -> dict[str, str]:
    output = {style_key: "" for style_key in STYLE_LABELS}
    for line in lines:
        if "：" not in line:
            continue
        prefix, value = line.split("：", 1)
        if "版本A" in prefix or "克制版" in prefix:
            output["restrained"] = value.strip()
        elif "版本B" in prefix or "冲击版" in prefix:
            output["impact"] = value.strip()
        elif "版本C" in prefix or "学术版" in prefix:
            output["academic"] = value.strip()
    return output


def parse_plan_document(plan_path: Path) -> dict:
    text = plan_path.read_text(encoding="utf-8")
    scene1 = extract_heading_block(text, "### 第一幕：茧房态", ["### 第二幕", "## 三、测试语料库"])
    scene2 = extract_heading_block(text, "### 第二幕：核查交互区", ["### 第三幕", "## 三、测试语料库"])
    scene3 = extract_heading_block(text, "### 第三幕：数据全景区", ["## 三、测试语料库", "## 四、使用建议"])
    corpus_section = extract_heading_block(text, "## 三、测试语料库", ["## 四、使用建议"])
    usage_section = extract_heading_block(text, "## 四、使用建议", [])

    styles: dict[str, dict] = {
        style_key: {
            "label": label,
            "floating_lines": [],
            "entry_prompt": "",
            "button_label": "",
            "panorama_intro": "",
            "slider_left": "",
            "slider_right": "",
            "warm_line": "",
            "closing": "",
            "status_labels": {},
        }
        for style_key, label in STYLE_LABELS.items()
    }

    for style_key, lines in parse_version_bullets(scene1).items():
        styles[style_key]["floating_lines"] = lines

    for row in extract_table_after_marker(scene2, "入口引导语"):
        style_key = STYLE_KEY_MAP.get(row.get("版本", ""))
        if style_key:
            styles[style_key]["entry_prompt"] = row.get("文案", "")

    for row in extract_table_after_marker(scene2, "按钮文案"):
        style_key = STYLE_KEY_MAP.get(row.get("版本", ""))
        if style_key:
            styles[style_key]["button_label"] = row.get("文案", "")

    for row in extract_table_after_marker(scene2, "结果状态标签"):
        state_key = {
            "支持": "supported",
            "部分支持": "partial",
            "不支持": "unsupported",
            "无相关记录": "uncovered",
        }.get(row.get("状态", ""))
        if not state_key:
            continue
        for version_name, style_key in STYLE_KEY_MAP.items():
            styles[style_key]["status_labels"][state_key] = row.get(version_name, "")

    for row in extract_table_after_marker(scene3, "入口引导语"):
        style_key = STYLE_KEY_MAP.get(row.get("版本", ""))
        if style_key:
            styles[style_key]["panorama_intro"] = row.get("文案", "")

    for row in extract_table_after_marker(scene3, "滑块标签"):
        style_key = STYLE_KEY_MAP.get(row.get("版本", ""))
        if style_key:
            styles[style_key]["slider_left"] = row.get("左端", "")
            styles[style_key]["slider_right"] = row.get("右端", "")

    warm_map = parse_version_line_items(extract_bullets_after_marker(scene3, "顽固暖色卡片提示"))
    closing_map = parse_version_line_items(extract_bullets_after_marker(scene3, "收尾文案"))
    for style_key in STYLE_LABELS:
        styles[style_key]["warm_line"] = warm_map.get(style_key, "")
        styles[style_key]["closing"] = closing_map.get(style_key, "")

    corpus_items: list[dict] = []
    current_category = ""
    for raw_line in corpus_section.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            current_category = line.replace("### ", "").strip()
            continue
        match = re.match(r"^(\d+)\.\s*【([^】]+)】(.+)$", line)
        if not match:
            continue
        label_full = match.group(2).strip()
        primary_label = label_full.split("，", 1)[0].split("/", 1)[0].strip()
        corpus_items.append(
            {
                "id": int(match.group(1)),
                "category": current_category,
                "label": primary_label,
                "label_full": label_full,
                "text": match.group(3).strip(),
            }
        )

    return {
        "styles": styles,
        "placeholders": extract_bullets_after_marker(scene2, "输入框placeholder"),
        "loading_texts": extract_bullets_after_marker(scene2, "加载态文案"),
        "result_notes": extract_bullets_after_marker(scene2, "结果附加说明"),
        "usage_notes": [line[2:].strip() for line in usage_section.splitlines() if line.strip().startswith("- ")],
        "corpus": corpus_items,
    }


def build_claim_records(rag_predictions: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, series in rag_predictions.iterrows():
        evidence = unique_preserve_order(
            [
                translate_evidence_text(item)
                for item in str(series.get("retrieved_texts", "")).split(" || ")
                if item.strip() and not item.strip().startswith("Disease summary for ")
            ]
        )[:4]

        predicted_label = str(series.get("predicted_label", "unknown"))
        records.append(
            {
                "sample_id": str(series["sample_id"]),
                "claim_text": translate_terms(str(series["claim_text"]).strip()),
                "relation_group": str(series["relation_group"]),
                "relation_group_label": RELATION_GROUP_LABELS.get(
                    str(series["relation_group"]), str(series["relation_group"])
                ),
                "false_type": str(series["false_type"]),
                "false_type_label": FALSE_TYPE_LABELS.get(str(series["false_type"]), str(series["false_type"])),
                "predicted_label": predicted_label,
                "predicted_label_text": VERDICT_LABELS.get(predicted_label, predicted_label),
                "top_score": round(float(series.get("top_score", 0.0)), 3),
                "x_name": translate_terms(str(series["x_name"])),
                "y_name": translate_terms(str(series["y_name"])),
                "display_relation_label": translate_relation_label(str(series["display_relation"])),
                "note": translate_note(str(series.get("note", ""))),
                "is_correct": str(series.get("is_correct", "")).lower() == "true" or bool(series.get("is_correct")),
                "evidence": evidence or ["当前没有额外展示的证据文本。"],
            }
        )
    return records


def pick_featured_claims(records: list[dict], limit: int) -> list[dict]:
    if len(records) <= limit:
        return records

    picked: list[dict] = []
    seen: set[str] = set()

    def add_candidates(items: list[dict], count: int) -> None:
        added = 0
        for item in items:
            if item["sample_id"] in seen:
                continue
            seen.add(item["sample_id"])
            picked.append(item)
            added += 1
            if added >= count:
                break

    for relation_group in ["disease_drug", "disease_phenotype", "disease_disease"]:
        group_items = [item for item in records if item["false_type"] == "real" and item["relation_group"] == relation_group]
        group_items.sort(key=lambda item: item["top_score"], reverse=True)
        add_candidates(group_items, 2)

    for false_type in ["fabricated", "polarity_flip", "hierarchy_error"]:
        false_items = [item for item in records if item["false_type"] == false_type]
        false_items.sort(key=lambda item: item["top_score"], reverse=True)
        add_candidates(false_items, 2)

    for item in sorted(records, key=lambda row: (not row["is_correct"], -row["top_score"])):
        if item["sample_id"] in seen:
            continue
        seen.add(item["sample_id"])
        picked.append(item)
        if len(picked) >= limit:
            break

    return picked[:limit]


def build_plan_risk_cards(plan_data: dict) -> list[dict]:
    corpus_map = {item["id"]: item for item in plan_data["corpus"]}
    return [
        {
            "title": "多跳推理",
            "summary": "需要跨越多条关系链才能完成核验。",
            "examples": [corpus_map[item_id]["text"] for item_id in (44, 45) if item_id in corpus_map],
        },
        {
            "title": "语义鸿沟",
            "summary": "口语热词与图谱术语之间存在表达落差。",
            "examples": [corpus_map[item_id]["text"] for item_id in (46, 47) if item_id in corpus_map],
        },
        {
            "title": "否定性问题",
            "summary": "“不存在任何关系”这类说法最容易诱发迎合式回答。",
            "examples": [corpus_map[item_id]["text"] for item_id in (48, 49) if item_id in corpus_map],
        },
        {
            "title": "虚构实体",
            "summary": "不存在的诊断名称会把模型引向更隐蔽的幻觉。",
            "examples": [corpus_map[item_id]["text"] for item_id in (50,) if item_id in corpus_map],
        },
    ]


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
                {
                    "label": "训练 / 验证 / 测试",
                    "value": f"{step2_dataset_summary['train_count']} / {step2_dataset_summary['valid_count']} / {step2_dataset_summary['test_count']}",
                },
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
                {"label": "真实 / 错误声明", "value": f"{step4_dataset_summary['real_samples']} / {step4_dataset_summary['false_samples']}"},
                {"label": "纯基线准确率", "value": format_percent(float(pure_row["accuracy"]))},
                {"label": "RAG 校验准确率", "value": format_percent(float(rag_row["accuracy"]))},
                {
                    "label": "层级错置识别率",
                    "value": format_percent(float(step4_simulated_summary["by_false_type"]["hierarchy_error"]["accuracy"])),
                },
            ],
        },
    ]


def build_family_distribution(step1_summary: dict) -> list[dict]:
    distribution = step1_summary.get("seed_family_distribution", {})
    total = max(sum(int(count) for count in distribution.values()), 1)
    return [
        {
            "name": name,
            "label": FAMILY_LABELS.get(name, name.replace("_", " ")),
            "count": int(count),
            "ratio": round(int(count) / total, 4),
        }
        for name, count in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    ]


def build_top_nodes(step1_summary: dict, limit: int = 8) -> list[dict]:
    return [{"name": translate_terms(row["node_name"]), "degree": int(row["degree"])} for row in step1_summary.get("top_disease_nodes", [])[:limit]]


def build_corpus_stats(plan_data: dict) -> dict:
    corpus_df = pd.DataFrame(plan_data["corpus"])
    if corpus_df.empty:
        return {"by_category": [], "by_label": []}
    by_category = [{"label": category, "count": int(count)} for category, count in corpus_df["category"].value_counts().items()]
    by_label = [
        {"label": label, "count": int((corpus_df["label"] == label).sum())}
        for label in RATING_ORDER
        if (corpus_df["label"] == label).any()
    ]
    return {"by_category": by_category, "by_label": by_label}


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
    plan_data = parse_plan_document(project_root / "plan.md")
    plan_data["styles"]["academic"]["slider_left"] = "纯基线模型"
    plan_data["styles"]["academic"]["slider_right"] = "图谱增强回答"

    all_claims = build_claim_records(rag_predictions)
    featured_claims = pick_featured_claims(all_claims, sample_limit)
    false_corpus_examples = [item for item in plan_data["corpus"] if item["label"] != "真实"]
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]

    return {
        "meta": {
            "title": "PrimeKG 精神健康信息茧房交互演示",
            "subtitle": "",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "styles": plan_data["styles"],
        "style_themes": STYLE_THEMES,
        "placeholders": plan_data["placeholders"],
        "loading_texts": plan_data["loading_texts"],
        "result_notes": plan_data["result_notes"],
        "usage_notes": plan_data["usage_notes"],
        "summary_tables": build_summary_tables(
            step1_summary=step1_summary,
            step2_dataset_summary=step2_dataset_summary,
            kge_df=kge_df,
            classification_df=classification_df,
            step4_dataset_summary=step4_dataset_summary,
            step4_simulated_summary=step4_simulated_summary,
            rag_summary_df=rag_summary_df,
        ),
        "panorama": {
            "baseline_accuracy": float(pure_row["accuracy"]),
            "baseline_hallucination_rate": 1.0 - float(pure_row["accuracy"]),
            "rag_accuracy": float(rag_row["accuracy"]),
            "rag_hallucination_rate": 1.0 - float(rag_row["accuracy"]),
            "unresolved_count": int(sum(1 for item in all_claims if item["predicted_label"] == "unknown")),
            "risk_cards": build_plan_risk_cards(plan_data),
        },
        "claims": all_claims,
        "featured_claims": featured_claims,
        "false_corpus_examples": false_corpus_examples,
        "corpus": plan_data["corpus"],
        "corpus_stats": build_corpus_stats(plan_data),
        "family_distribution": build_family_distribution(step1_summary),
        "top_nodes": build_top_nodes(step1_summary),
    }


def render_demo_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    template = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #070c10;
      --bg-2: #0d141a;
      --panel: rgba(10, 17, 24, 0.74);
      --panel-strong: rgba(8, 13, 19, 0.88);
      --line: rgba(255, 255, 255, 0.08);
      --text: #ecf7ff;
      --muted: #93a7b6;
      --hot: #ff8b5f;
      --hot-soft: rgba(255, 139, 95, 0.16);
      --cold: #8ddcff;
      --cold-soft: rgba(141, 220, 255, 0.16);
      --gold: #f6c46d;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
      --radius-xl: 34px;
      --radius-lg: 26px;
      --radius-md: 20px;
      --sans: "Microsoft YaHei UI", "PingFang SC", "Noto Sans SC", sans-serif;
      --serif: "STZhongsong", "STSong", "SimSun", serif;
      --accent: #8ddcff;
      --accent-soft: rgba(141, 220, 255, 0.16);
      --accent-glow: #a8f0ff;
      --mouse-x: 50%;
      --mouse-y: 50%;
    }

    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 12% 18%, rgba(255, 108, 66, 0.16), transparent 22%),
        radial-gradient(circle at 82% 12%, rgba(141, 220, 255, 0.16), transparent 24%),
        radial-gradient(circle at var(--mouse-x) var(--mouse-y), rgba(168, 240, 255, 0.12), transparent 22%),
        linear-gradient(180deg, #06090d 0%, #0a1016 42%, #081017 100%);
      overflow-x: hidden;
    }

    #fieldCanvas {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: -1;
      pointer-events: none;
      opacity: 0.72;
      mix-blend-mode: screen;
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: -2;
    }

    body::before {
      background:
        repeating-linear-gradient(
          115deg,
          rgba(255, 255, 255, 0.016) 0,
          rgba(255, 255, 255, 0.016) 1px,
          transparent 1px,
          transparent 10px
        );
      mix-blend-mode: screen;
      opacity: 0.5;
    }

    body::after {
      background:
        radial-gradient(circle at 50% -10%, rgba(255, 189, 111, 0.08), transparent 35%),
        radial-gradient(circle at 50% 120%, rgba(110, 201, 255, 0.08), transparent 30%);
    }

    .page {
      width: min(1440px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 18px 0 56px;
    }

    .topbar {
      position: sticky;
      top: 12px;
      z-index: 30;
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 16px;
      align-items: center;
      margin-bottom: 18px;
      padding: 14px 18px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      background: rgba(7, 11, 16, 0.72);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }

    .brand strong {
      font-size: 1rem;
      letter-spacing: 0.08em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .brand span {
      color: var(--muted);
      font-size: 0.82rem;
    }

    .switcher,
    .quick-actions,
    .chip-row,
    .filter-row,
    .meta-row,
    .result-top,
    .stats-row,
    .library-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    button,
    input,
    textarea {
      font: inherit;
    }

    .pill,
    .ghost-btn,
    .chip,
    .filter-chip,
    .micro-btn,
    .suggestion-pill {
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      cursor: pointer;
      transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }

    .pill,
    .ghost-btn,
    .chip,
    .filter-chip,
    .micro-btn {
      padding: 10px 15px;
    }

    .pill:hover,
    .ghost-btn:hover,
    .chip:hover,
    .filter-chip:hover,
    .micro-btn:hover,
    .suggestion-pill:hover {
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.14);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
    }

    .pill.is-active,
    .filter-chip.is-active {
      background: linear-gradient(135deg, var(--accent-soft), rgba(255, 255, 255, 0.06));
      border-color: rgba(255, 255, 255, 0.18);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.05), 0 0 26px rgba(168, 240, 255, 0.16);
    }

    .ghost-btn.accent,
    .micro-btn.accent {
      background: linear-gradient(135deg, rgba(255, 133, 90, 0.18), rgba(255, 208, 126, 0.12));
      border-color: rgba(255, 196, 106, 0.18);
    }

    .scene {
      position: relative;
      margin-top: 18px;
      border-radius: var(--radius-xl);
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(8, 12, 18, 0.72);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      overflow: hidden;
      opacity: 0;
      transform: translateY(36px);
      transition: opacity 650ms ease, transform 650ms cubic-bezier(.2,.8,.2,1);
    }

    .scene.is-visible {
      opacity: 1;
      transform: translateY(0);
    }

    .scene::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(120deg, rgba(255,255,255,0.03), transparent 26%, transparent 72%, rgba(255,255,255,0.02)),
        radial-gradient(circle at 84% 14%, rgba(255, 144, 88, 0.08), transparent 18%);
      pointer-events: none;
    }

    .scene-head {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      padding: 22px 24px 0;
    }

    .scene-title {
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(1.8rem, 3vw, 2.8rem);
      line-height: 1.08;
      letter-spacing: 0.02em;
    }

    .scene-intro {
      margin: 10px 0 0;
      max-width: 44rem;
      color: var(--muted);
      line-height: 1.7;
      font-size: 0.96rem;
    }

    .hero-grid {
      position: relative;
      display: grid;
      grid-template-columns: 1.02fr 0.98fr;
      gap: 18px;
      padding: 22px 24px 28px;
      min-height: 760px;
    }

    .hero-main,
    .scanner-panel,
    .result-panel,
    .compare-panel,
    .summary-panel,
    .risk-panel,
    .library-panel {
      position: relative;
      border-radius: var(--radius-lg);
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: linear-gradient(180deg, rgba(8, 14, 20, 0.88), rgba(6, 10, 15, 0.92));
      overflow: hidden;
    }

    .hero-main {
      padding: 30px;
      min-height: 100%;
      display: grid;
      align-content: space-between;
      isolation: isolate;
      animation: ambientLift 7s ease-in-out infinite;
    }

    .hero-main::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        radial-gradient(circle at 12% 16%, rgba(255, 128, 88, 0.14), transparent 24%),
        radial-gradient(circle at 88% 18%, rgba(141, 220, 255, 0.14), transparent 22%),
        linear-gradient(135deg, rgba(255,255,255,0.03), transparent 40%);
      opacity: 0.9;
      pointer-events: none;
    }

    .hero-copy {
      position: relative;
      z-index: 1;
      max-width: 36rem;
    }

    .hero-copy h1 {
      margin: 16px 0 12px;
      font-family: var(--serif);
      font-size: clamp(2.6rem, 5vw, 4.8rem);
      line-height: 0.98;
      letter-spacing: -0.03em;
    }

    .hero-copy p {
      margin: 0;
      color: #d3e5f3;
      line-height: 1.8;
      max-width: 34rem;
    }

    .stats-row {
      margin-top: 24px;
    }

    .hero-stat,
    .metric-box,
    .summary-card,
    .library-stat {
      min-width: 0;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      backdrop-filter: blur(10px);
    }

    .hero-stat span,
    .metric-box span,
    .library-stat span {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 8px;
      letter-spacing: 0.06em;
    }

    .hero-stat strong,
    .metric-box strong,
    .library-stat strong {
      font-family: var(--serif);
      font-size: 1.3rem;
    }

    .hero-stage {
      position: relative;
      min-height: 100%;
      overflow: hidden;
      border-radius: var(--radius-lg);
      background:
        radial-gradient(circle at 50% 50%, rgba(168, 240, 255, 0.12), transparent 18%),
        radial-gradient(circle at 50% 50%, rgba(255, 152, 104, 0.10), transparent 26%),
        linear-gradient(180deg, rgba(6, 10, 16, 0.94), rgba(7, 12, 18, 0.98));
      border: 1px solid rgba(255, 255, 255, 0.08);
      isolation: isolate;
    }

    .hero-stage::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, transparent 0, rgba(255,255,255,0.03) 50%, transparent 100%),
        repeating-linear-gradient(
          180deg,
          rgba(255,255,255,0.015) 0,
          rgba(255,255,255,0.015) 1px,
          transparent 1px,
          transparent 14px
        );
      mix-blend-mode: screen;
      opacity: 0.45;
      pointer-events: none;
    }

    .cocoon-center {
      position: absolute;
      inset: 50% auto auto 50%;
      width: 240px;
      height: 240px;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      background:
        radial-gradient(circle, rgba(255,255,255,0.08) 0, rgba(141,220,255,0.06) 38%, transparent 72%);
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.05),
        0 0 60px rgba(168, 240, 255, 0.12),
        0 0 120px rgba(255, 143, 100, 0.10);
      display: grid;
      place-items: center;
      z-index: 1;
    }

    .cocoon-core {
      width: 122px;
      height: 122px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      text-align: center;
      font-family: var(--serif);
      font-size: 1rem;
      line-height: 1.55;
      border: 1px solid rgba(255,255,255,0.12);
      background: radial-gradient(circle, rgba(8, 14, 20, 0.9), rgba(8, 14, 20, 0.42));
      box-shadow: 0 0 42px rgba(0, 0, 0, 0.36);
    }

    .cocoon-wall {
      position: absolute;
      inset: 0;
      overflow: hidden;
    }

    .floating-shard {
      position: absolute;
      width: min(42vw, 220px);
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: rgba(244, 249, 255, 0.92);
      line-height: 1.6;
      font-size: 0.95rem;
      letter-spacing: 0.02em;
      backdrop-filter: blur(8px);
      transform: translate3d(calc(var(--tx) + var(--px, 0px)), calc(var(--ty) + var(--py, 0px)), 0) rotate(var(--rot));
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
      animation: floatShard var(--dur) ease-in-out infinite;
      animation-delay: var(--delay);
      opacity: var(--alpha);
      transition: filter 220ms ease, border-color 220ms ease;
    }

    .floating-shard.hot {
      background: rgba(255, 140, 96, 0.08);
      border-color: rgba(255, 163, 120, 0.14);
    }

    .floating-shard.cold {
      background: rgba(141, 220, 255, 0.08);
      border-color: rgba(141, 220, 255, 0.14);
    }

    .floating-shard strong {
      display: block;
      margin-bottom: 6px;
      color: rgba(255,255,255,0.52);
      font-size: 0.72rem;
      letter-spacing: 0.18em;
    }

    .hero-stage:hover .floating-shard {
      filter: brightness(1.08);
      border-color: rgba(255,255,255,0.14);
    }

    .fracture-ring {
      position: absolute;
      inset: 50% auto auto 50%;
      width: 460px;
      height: 460px;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: inset 0 0 80px rgba(255,255,255,0.02);
      animation: rotateRing 26s linear infinite;
      opacity: 0.7;
    }

    .fracture-ring::before,
    .fracture-ring::after {
      content: "";
      position: absolute;
      inset: 16%;
      border-radius: 50%;
      border: 1px dashed rgba(255,255,255,0.08);
    }

    .fracture-ring::after {
      inset: 31%;
      border-style: solid;
      border-color: rgba(255, 255, 255, 0.05);
    }

    .verifier-grid,
    .panorama-grid,
    .library-grid {
      display: grid;
      gap: 18px;
      padding: 22px 24px 28px;
    }

    .verifier-grid {
      grid-template-columns: 0.96fr 1.04fr;
    }

    .scanner-panel,
    .result-panel,
    .compare-panel,
    .summary-panel,
    .risk-panel,
    .library-panel {
      padding: 22px;
    }

    .panel-label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
    }

    .scanner-shell {
      position: relative;
      margin-top: 16px;
      padding: 18px;
      border-radius: 26px;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        radial-gradient(circle at 50% 0%, rgba(141,220,255,0.12), transparent 34%);
      overflow: hidden;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
      transition: transform 220ms ease, box-shadow 220ms ease;
    }

    .scanner-shell.is-focus {
      transform: translateY(-2px);
      box-shadow:
        inset 0 0 0 1px rgba(255,255,255,0.04),
        0 0 26px rgba(168, 240, 255, 0.08);
    }

    .scanner-shell.is-active::after {
      content: "";
      position: absolute;
      inset: -25% 0 auto;
      height: 42%;
      background: linear-gradient(180deg, transparent, rgba(168, 240, 255, 0.24), transparent);
      animation: scanDown 1s ease;
      pointer-events: none;
    }

    .scanner-shell.is-active::before {
      content: "";
      position: absolute;
      inset: 50% auto auto 50%;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      border: 1px solid rgba(168, 240, 255, 0.36);
      transform: translate(-50%, -50%);
      animation: shockwave 900ms ease-out forwards;
      pointer-events: none;
    }

    .query-input {
      width: 100%;
      min-height: 164px;
      border: 0;
      resize: vertical;
      background: transparent;
      color: var(--text);
      outline: none;
      line-height: 1.75;
      font-size: 1rem;
    }

    .query-input::placeholder {
      color: rgba(212, 227, 238, 0.36);
    }

    .loading-strip {
      display: grid;
      gap: 10px;
      margin-top: 14px;
      color: var(--accent-glow);
    }

    .loading-strip strong {
      font-size: 0.9rem;
      letter-spacing: 0.06em;
    }

    .loading-track {
      position: relative;
      height: 6px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.05);
    }

    .loading-track::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 -35%;
      width: 35%;
      border-radius: inherit;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      animation: drift 2.2s linear infinite;
    }

    .quick-zone {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }

    .quick-zone h3,
    .risk-panel h3,
    .summary-panel h3,
    .library-panel h3,
    .compare-panel h3 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.04em;
    }

    .quick-zone p,
    .compare-caption,
    .panel-note,
    .library-note,
    .summary-note,
    .risk-note {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 0.92rem;
    }

    .chip-row {
      align-items: stretch;
    }

    .chip,
    .suggestion-pill {
      padding: 10px 14px;
      text-align: left;
      line-height: 1.55;
    }

    .suggestions-cloud {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }

    .suggestion-pill {
      min-height: 72px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
    }

    .suggestion-pill span {
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 6px;
    }

    .result-panel {
      isolation: isolate;
      min-height: 680px;
      background:
        radial-gradient(circle at 12% 14%, rgba(255, 141, 98, 0.10), transparent 24%),
        radial-gradient(circle at 84% 16%, rgba(141, 220, 255, 0.10), transparent 20%),
        linear-gradient(180deg, rgba(8, 14, 20, 0.9), rgba(6, 9, 14, 0.98));
    }

    .result-panel::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(130deg, rgba(255,255,255,0.03), transparent 32%),
        linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.02) 100%);
      pointer-events: none;
    }

    .result-panel.is-active .result-flare {
      opacity: 1;
      transform: scale(1.04);
    }

    .result-panel::after {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(120deg, transparent 0 28%, rgba(255,255,255,0.06) 30%, transparent 32%, transparent 58%, rgba(255,255,255,0.05) 60%, transparent 62%, transparent),
        linear-gradient(180deg, transparent, rgba(168, 240, 255, 0.06), transparent);
      opacity: 0;
      transition: opacity 280ms ease;
      pointer-events: none;
      mix-blend-mode: screen;
    }

    .result-panel.is-active::after {
      opacity: 1;
      animation: crackle 900ms ease-out;
    }

    .result-flare {
      position: absolute;
      inset: auto 8% -10% auto;
      width: 340px;
      height: 340px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(168, 240, 255, 0.14), transparent 62%);
      filter: blur(10px);
      opacity: 0.5;
      transition: transform 500ms ease, opacity 500ms ease;
      pointer-events: none;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.05);
      font-size: 0.82rem;
      letter-spacing: 0.05em;
    }

    .badge.supported { background: rgba(110, 224, 171, 0.14); color: #baffdf; }
    .badge.partial { background: rgba(255, 212, 123, 0.14); color: #ffe5a6; }
    .badge.unsupported { background: rgba(255, 129, 110, 0.16); color: #ffd0c5; }
    .badge.uncovered { background: rgba(146, 169, 188, 0.14); color: #d8e7f2; }
    .badge.soft { color: var(--muted); }

    .verdict-lockup {
      margin-top: 18px;
      padding: 24px 0 18px;
      border-top: 1px solid rgba(255,255,255,0.08);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .verdict-word {
      font-family: var(--serif);
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: 1;
      letter-spacing: -0.02em;
      text-shadow: 0 0 28px rgba(168, 240, 255, 0.14);
      transform-origin: left center;
    }

    .result-panel.is-revealed .verdict-word {
      animation: pulseIn 540ms cubic-bezier(.16,.84,.26,1.2);
    }

    .scene.is-visible .hero-stage,
    .scene.is-visible .scanner-panel,
    .scene.is-visible .result-panel,
    .scene.is-visible .compare-panel,
    .scene.is-visible .summary-panel,
    .scene.is-visible .risk-panel,
    .scene.is-visible .library-panel {
      animation: riseIn 560ms both;
    }

    .verdict-sub {
      margin-top: 10px;
      color: #e2edf6;
      line-height: 1.7;
      font-size: 1rem;
      max-width: 40rem;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .result-box {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
    }

    .result-box span {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }

    .result-box strong {
      display: block;
      font-size: 0.98rem;
      line-height: 1.7;
    }

    .evidence-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.85rem;
      letter-spacing: 0.08em;
    }

    .evidence-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }

    .evidence-item {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(135deg, rgba(141, 220, 255, 0.08), rgba(255,255,255,0.03));
      line-height: 1.65;
      opacity: 0;
      transform: translateY(8px);
      animation: riseIn 420ms forwards;
    }

    .panorama-grid {
      grid-template-columns: 1.08fr 0.92fr;
      align-items: start;
    }

    .compare-panel {
      display: grid;
      gap: 16px;
    }

    .compare-shell {
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.08);
      background:
        linear-gradient(90deg, rgba(255, 137, 96, 0.12), rgba(141, 220, 255, 0.12)),
        rgba(255,255,255,0.02);
    }

    .compare-mode {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-family: var(--serif);
      font-size: 1.6rem;
    }

    .compare-mode span {
      color: var(--muted);
      font-family: var(--sans);
      font-size: 0.86rem;
      letter-spacing: 0.06em;
    }

    .range-wrap {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }

    .range-labels {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.84rem;
    }

    .range-input {
      width: 100%;
      accent-color: var(--accent-glow);
    }

    .metric-grid,
    .summary-stack,
    .risk-grid,
    .library-stats,
    .library-list,
    .usage-list {
      display: grid;
      gap: 12px;
    }

    .metric-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 12px;
    }

    .metric-box {
      min-height: 110px;
    }

    .summary-card {
      padding: 18px;
    }

    .summary-card table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }

    .summary-card td {
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 0.92rem;
    }

    .summary-card td:first-child {
      color: var(--muted);
      width: 44%;
    }

    .summary-card td:last-child {
      text-align: right;
      font-family: var(--serif);
      font-size: 1.04rem;
      letter-spacing: 0.02em;
    }

    .summary-card tr:last-child td {
      border-bottom: 0;
    }

    .risk-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 16px;
    }

    .risk-card {
      padding: 18px;
      border-radius: 20px;
      border: 1px solid rgba(255, 167, 124, 0.12);
      background:
        linear-gradient(180deg, rgba(255, 140, 96, 0.10), rgba(255,255,255,0.03)),
        rgba(255,255,255,0.03);
    }

    .risk-card strong {
      display: block;
      font-size: 1rem;
      margin-bottom: 8px;
    }

    .risk-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .risk-card ul {
      margin: 12px 0 0;
      padding-left: 18px;
      color: #e5eff7;
      line-height: 1.7;
    }

    .closing-line {
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: #f6fbff;
      font-family: var(--serif);
      font-size: 1.02rem;
      line-height: 1.8;
    }

    .library-grid {
      grid-template-columns: 0.96fr 1.04fr;
      align-items: start;
    }

    .library-note {
      margin-top: 10px;
    }

    .library-stats {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 16px;
    }

    .library-list {
      max-height: 780px;
      overflow: auto;
      padding-right: 4px;
    }

    .library-item {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      display: grid;
      gap: 12px;
    }

    .library-item h4 {
      margin: 0;
      font-size: 1rem;
      line-height: 1.7;
    }

    .usage-list {
      margin-top: 16px;
    }

    .usage-item {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: #dceaf5;
      line-height: 1.7;
    }

    .footer-mark {
      margin-top: 18px;
      text-align: right;
      color: rgba(255,255,255,0.34);
      font-size: 0.8rem;
      letter-spacing: 0.08em;
    }

    @keyframes floatShard {
      0%, 100% { transform: translate3d(var(--tx), calc(var(--ty) - 6px), 0) rotate(var(--rot)); }
      50% { transform: translate3d(var(--tx), calc(var(--ty) + 12px), 0) rotate(calc(var(--rot) * -1)); }
    }

    @keyframes rotateRing {
      from { transform: translate(-50%, -50%) rotate(0deg); }
      to { transform: translate(-50%, -50%) rotate(360deg); }
    }

    @keyframes drift {
      from { transform: translateX(0); }
      to { transform: translateX(390%); }
    }

    @keyframes scanDown {
      from { transform: translateY(-110%); opacity: 0; }
      30% { opacity: 1; }
      to { transform: translateY(240%); opacity: 0; }
    }

    @keyframes shockwave {
      0% { opacity: 0.9; width: 32px; height: 32px; }
      100% { opacity: 0; width: 720px; height: 720px; }
    }

    @keyframes pulseIn {
      0% { opacity: 0; transform: scale(0.92) translateY(8px); }
      100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    @keyframes riseIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes ambientLift {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-4px); }
    }

    @keyframes crackle {
      0% { opacity: 0; transform: translateY(12px); }
      30% { opacity: 0.8; }
      100% { opacity: 0; transform: translateY(-12px); }
    }

    @media (max-width: 1100px) {
      .topbar,
      .hero-grid,
      .verifier-grid,
      .panorama-grid,
      .library-grid {
        grid-template-columns: 1fr;
      }

      .hero-grid {
        min-height: auto;
      }
    }

    @media (max-width: 760px) {
      .page {
        width: min(100vw, calc(100vw - 16px));
      }

      .topbar {
        position: relative;
        top: 0;
        border-radius: 26px;
        grid-template-columns: 1fr;
      }

      .scene-head,
      .hero-grid,
      .verifier-grid,
      .panorama-grid,
      .library-grid {
        padding-left: 16px;
        padding-right: 16px;
      }

      .hero-main,
      .scanner-panel,
      .result-panel,
      .compare-panel,
      .summary-panel,
      .risk-panel,
      .library-panel {
        padding: 18px;
      }

      .stats-row,
      .metric-grid,
      .result-grid,
      .library-stats,
      .risk-grid,
      .family-legend,
      .suggestions-cloud {
        grid-template-columns: 1fr;
      }

      .cocoon-center {
        width: 170px;
        height: 170px;
      }

      .cocoon-core {
        width: 98px;
        height: 98px;
        font-size: 0.88rem;
      }

      .fracture-ring {
        width: 300px;
        height: 300px;
      }

      .floating-shard {
        width: min(70vw, 210px);
      }
    }
  </style>
</head>
<body>
  <canvas id="fieldCanvas" aria-hidden="true"></canvas>
  <div class="page" id="pageRoot">
    <header class="topbar">
      <div class="brand">
        <strong>PrimeKG 精神健康信息茧房交互演示</strong>
        <span>生成时间：__GENERATED_AT__</span>
      </div>
      <div class="switcher">
        <span class="pill is-active">01 围住</span>
        <span class="pill">02 撕开</span>
        <span class="pill">03 露出</span>
      </div>
      <div class="quick-actions">
        <button class="ghost-btn accent" type="button" id="randomFalseBtn">随机抽一条危险说法</button>
      </div>
    </header>

    <section class="scene">
      <div class="scene-head">
        <div>
          <h1 class="scene-title">先被短视频包围，再决定你该相信什么。</h1>
        </div>
      </div>
      <div class="hero-grid">
        <article class="hero-main">
          <div class="hero-copy">
            <div class="panel-label">信息茧房 / 算法推送 / 心理健康话术</div>
            <h1>你刷到的每一句，<br>都可能比证据先一步进入你脑海。</h1>
            <p>先被围住，再把它撕开。</p>
            <div class="stats-row">
              <div class="hero-stat"><span>实验声明</span><strong id="claimCount"></strong></div>
              <div class="hero-stat"><span>种子疾病</span><strong id="seedCount"></strong></div>
            <div class="hero-stat"><span>图谱校验准确率</span><strong id="ragAcc"></strong></div>
            </div>
          </div>
          <div class="chip-row" id="heroActions">
            <button class="micro-btn accent" type="button" id="enterAct2Btn">进入第二幕</button>
            <button class="micro-btn" type="button" id="sampleMythBtn">先试一句</button>
          </div>
        </article>
        <article class="hero-stage" id="heroStage">
          <div class="fracture-ring"></div>
          <div class="cocoon-center">
            <div class="cocoon-core">先怀疑<br>再核查</div>
          </div>
          <div class="cocoon-wall" id="cocoonWall"></div>
        </article>
      </div>
    </section>

    <section class="scene" id="verifyScene" hidden>
      <div class="scene-head">
        <div>
          <h2 class="scene-title">把一句话丢进来，看它能不能撑住。</h2>
          <p class="scene-intro" id="entryPrompt"></p>
        </div>
      </div>
      <div class="verifier-grid">
        <article class="scanner-panel">
          <div class="panel-label">核查入口</div>
          <div class="scanner-shell" id="scannerShell">
            <textarea class="query-input" id="queryInput" spellcheck="false"></textarea>
          </div>
          <div class="loading-strip">
            <strong id="loadingLine"></strong>
            <div class="loading-track"></div>
          </div>
          <div class="chip-row" style="margin-top: 16px;">
            <button class="ghost-btn accent" type="button" id="verifyButton">启动核查</button>
            <button class="ghost-btn" type="button" id="clearInputBtn">换一句再试</button>
          </div>
          <div class="quick-zone">
            <div>
              <h3>猜你会想试</h3>
              <p class="panel-note">词越热，越要小心。</p>
            </div>
            <div class="suggestions-cloud" id="suggestionList"></div>
          </div>
          <div class="quick-zone">
            <div>
              <h3>随手扔一句</h3>
              <p class="panel-note">随手挑一句。</p>
            </div>
            <div class="chip-row" id="placeholderChips"></div>
          </div>
          <div class="quick-zone">
            <div>
              <h3>深一点的样例</h3>
              <p class="panel-note">往更难的地方试。</p>
            </div>
            <div class="chip-row" id="featuredClaims"></div>
          </div>
        </article>

        <article class="result-panel" id="resultPanel">
          <div class="result-flare"></div>
          <div class="panel-label">证据显影区</div>
          <div class="result-top" id="resultTop"></div>
          <div class="verdict-lockup">
            <div class="verdict-word" id="verdictWord">还没裂开</div>
            <div class="verdict-sub" id="resultTitle">先丢一句进来。</div>
          </div>
          <div class="result-grid">
            <div class="result-box"><span>余波</span><strong id="resultNote"></strong></div>
            <div class="result-box"><span>命中关系</span><strong id="resultRelation"></strong></div>
            <div class="result-box"><span>实体对</span><strong id="resultEntities"></strong></div>
            <div class="result-box"><span>匹配分数</span><strong id="resultScore"></strong></div>
          </div>
          <div class="evidence-head">
            <span>证据显影</span>
            <span id="evidenceHint">命中结果会在这里逐条展开</span>
          </div>
          <div class="evidence-list" id="evidenceList"></div>
          <div class="chip-row" style="margin-top:18px;">
            <button class="ghost-btn accent" type="button" id="revealAct3Btn" hidden>进入第三幕</button>
          </div>
        </article>
      </div>
    </section>

    <section class="scene" id="panoramaScene" hidden>
      <div class="scene-head">
        <div>
          <h2 class="scene-title">再往下看，它为什么会错。</h2>
          <p class="scene-intro" id="panoramaIntro"></p>
        </div>
      </div>
      <div class="panorama-grid">
        <article class="compare-panel">
          <div>
            <h3>纯基线 与 图谱校验</h3>
            <p class="compare-caption">拖一下就行。</p>
          </div>
          <div class="compare-shell">
            <div class="compare-mode">
              <div id="compareMode">图谱增强回答</div>
              <span id="warmLine"></span>
            </div>
            <div class="range-wrap">
              <div class="range-labels">
                <span id="sliderLeft"></span>
                <span id="sliderRight"></span>
              </div>
              <input class="range-input" id="compareSlider" type="range" min="0" max="100" step="100" value="100">
            </div>
            <div class="metric-grid">
              <div class="metric-box"><span>准确率</span><strong id="compareAccuracy"></strong></div>
              <div class="metric-box"><span>幻觉率</span><strong id="compareHallucination"></strong></div>
              <div class="metric-box"><span>未核实条数</span><strong id="compareUnresolved"></strong></div>
            </div>
          </div>
          <div class="closing-line" id="closingLine"></div>
        </article>

        <article class="summary-panel">
          <div>
            <h3>关键指标</h3>
            <p class="summary-note">只看结果。</p>
          </div>
          <div class="summary-stack" id="summaryTables"></div>
        </article>
      </div>
      <div class="panorama-grid" style="padding-top: 0;">
        <article class="risk-panel" style="grid-column: 1 / -1;">
          <div>
            <h3>顽固失效区</h3>
            <p class="risk-note">有些地方，光照不进去。</p>
          </div>
          <div class="risk-grid" id="riskCards"></div>
        </article>
      </div>
    </section>
  </div>

  <script>
    const payload = __PAYLOAD_JSON__;
    const state = {
      style: "impact",
      slider: 100,
      category: "all",
      label: "all",
      loadingIndex: 0,
      lastStatus: "unknown"
    };

    const qs = (id) => document.getElementById(id);
    const normalize = (text) => String(text || "")
      .toLowerCase()
      .replace(/[^\u4e00-\u9fff\w]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const tokenize = (text) => normalize(text).match(/[\u4e00-\u9fff]+|[a-z0-9_-]+/g) || [];
    const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

    const statusFallbacks = {
      supported: "图谱支持",
      partial: "部分成立",
      unsupported: "图谱不支持",
      uncovered: "图谱未覆盖",
      unknown: "待判断"
    };

    const verdictLabels = {
      supported: "支持",
      partial: "部分支持",
      unsupported: "不支持",
      uncovered: "无相关记录",
      unknown: "未知"
    };

    const familyPalette = ["#7dd3fc", "#f7b267", "#86efac", "#f9a8d4", "#c4b5fd", "#fdba74", "#67e8f9", "#a7f3d0", "#fca5a5", "#eab308"];

    function applyStyleTheme() {
      const theme = payload.style_themes[state.style] || payload.style_themes.academic;
      document.documentElement.style.setProperty("--accent", theme.accent);
      document.documentElement.style.setProperty("--accent-soft", theme.accent_soft);
      document.documentElement.style.setProperty("--accent-glow", theme.glow);
    }

    function createChip(label, handler, extraClass = "") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `chip ${extraClass}`.trim();
      button.textContent = label;
      button.addEventListener("click", handler);
      return button;
    }

    function renderHero() {
      const style = payload.styles[state.style];
      qs("seedCount").textContent = payload.summary_tables[0].rows[0].value;
      qs("claimCount").textContent = String(payload.claims.length);
      qs("ragAcc").textContent = formatPercent(payload.panorama.rag_accuracy);

      const wall = qs("cocoonWall");
      wall.innerHTML = "";
      const positions = [
        { x: "4%", y: "8%", rot: "-6deg", kind: "hot" },
        { x: "56%", y: "6%", rot: "7deg", kind: "cold" },
        { x: "8%", y: "63%", rot: "4deg", kind: "hot" },
        { x: "60%", y: "68%", rot: "-7deg", kind: "cold" },
        { x: "22%", y: "28%", rot: "-9deg", kind: "hot" },
        { x: "66%", y: "34%", rot: "6deg", kind: "cold" },
      ];

      style.floating_lines.forEach((line, index) => {
        const card = document.createElement("article");
        const pos = positions[index % positions.length];
        card.className = `floating-shard ${pos.kind}`;
        card.style.left = pos.x;
        card.style.top = pos.y;
        card.style.setProperty("--tx", "0px");
        card.style.setProperty("--ty", "0px");
        card.style.setProperty("--rot", pos.rot);
        card.style.setProperty("--delay", `${index * 0.38}s`);
        card.style.setProperty("--dur", `${9 + index * 0.8}s`);
        card.style.setProperty("--alpha", `${0.78 - (index % 3) * 0.1}`);
        card.style.setProperty("--px", "0px");
        card.style.setProperty("--py", "0px");
        card.dataset.depth = String(8 + index * 2);
        card.innerHTML = `<strong>碎片 ${String(index + 1).padStart(2, "0")}</strong>${line}`;
        wall.appendChild(card);
      });
    }

    function initHeroParallax() {
      const stage = qs("heroStage");
      if (!stage || stage.dataset.bound === "true") return;
      stage.dataset.bound = "true";
      stage.addEventListener("mousemove", (event) => {
        const rect = stage.getBoundingClientRect();
        const dx = (event.clientX - rect.left) / rect.width - 0.5;
        const dy = (event.clientY - rect.top) / rect.height - 0.5;
        stage.querySelectorAll(".floating-shard").forEach((node) => {
          const depth = Number(node.dataset.depth || 10);
          node.style.setProperty("--px", `${dx * depth * 4}px`);
          node.style.setProperty("--py", `${dy * depth * 4}px`);
        });
      });
      stage.addEventListener("mouseleave", () => {
        stage.querySelectorAll(".floating-shard").forEach((node) => {
          node.style.setProperty("--px", "0px");
          node.style.setProperty("--py", "0px");
        });
      });
    }

    function renderVerifierCopy() {
      const style = payload.styles[state.style];
      qs("entryPrompt").textContent = style.entry_prompt || "输入待核验陈述";
      qs("verifyButton").textContent = style.button_label || "启动核查";
      qs("loadingLine").textContent = payload.loading_texts[state.loadingIndex % Math.max(payload.loading_texts.length, 1)] || "正在图谱中检索…";
      const input = qs("queryInput");
      if (!input.value.trim()) {
        input.placeholder = payload.placeholders[state.loadingIndex % Math.max(payload.placeholders.length, 1)] || "输入一句你刷到的说法";
      }

      const chips = qs("placeholderChips");
      chips.innerHTML = "";
      payload.placeholders.forEach((text) => {
        chips.appendChild(createChip(text, () => {
          input.value = text;
          renderSuggestions();
          runVerification(false);
        }));
      });

      const featured = qs("featuredClaims");
      featured.innerHTML = "";
      payload.featured_claims.forEach((claim) => {
        const short = claim.claim_text.length > 26 ? claim.claim_text.slice(0, 26) + "…" : claim.claim_text;
        featured.appendChild(createChip(short, () => {
          input.value = claim.claim_text;
          renderSuggestions();
          runVerification(false);
        }, "accent"));
      });
    }

    function buildSuggestionItems(query) {
      const queryNorm = normalize(query);
      if (!queryNorm) {
        const fromCorpus = payload.corpus.slice(0, 6).map((item, index) => ({
          label: item.text,
          hint: `${item.category} / ${item.label_full}`,
          value: item.text,
          score: 100 - index
        }));
        const fromClaims = payload.featured_claims.slice(0, 2).map((claim, index) => ({
          label: claim.claim_text,
          hint: `${claim.relation_group_label} / ${claim.false_type_label}`,
          value: claim.claim_text,
          score: 40 - index
        }));
        return [...fromCorpus, ...fromClaims];
      }
      const items = [];

      payload.featured_claims.forEach((claim) => {
        const score = scoreClaim(query, claim);
        if (!queryNorm || score > 2.2) {
          items.push({
            label: claim.claim_text,
            hint: `${claim.relation_group_label} / ${claim.false_type_label}`,
            value: claim.claim_text,
            score
          });
        }
      });

      payload.corpus.forEach((item) => {
        const score = scoreCorpus(query, item);
        if (!queryNorm || score > 1.4) {
          items.push({
            label: item.text,
            hint: `${item.category} / ${item.label_full}`,
            value: item.text,
            score
          });
        }
      });

      return items
        .sort((a, b) => b.score - a.score)
        .filter((item, index, array) => array.findIndex((entry) => entry.label === item.label) === index)
        .slice(0, 8);
    }

    function renderSuggestions() {
      const root = qs("suggestionList");
      root.innerHTML = "";
      buildSuggestionItems(qs("queryInput").value).forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion-pill";
        button.innerHTML = `${item.label}<span>${item.hint}</span>`;
        button.addEventListener("click", () => {
          qs("queryInput").value = item.value;
          renderSuggestions();
        });
        root.appendChild(button);
      });
    }

    function scoreClaim(query, claim) {
      const queryNorm = normalize(query);
      if (!queryNorm) return 0;
      const queryTokens = tokenize(queryNorm);
      const fields = [claim.claim_text, claim.x_name, claim.y_name, claim.display_relation_label, claim.relation_group_label, claim.false_type_label];
      let score = 0;
      fields.forEach((field) => {
        const fieldNorm = normalize(field);
        if (!fieldNorm) return;
        if (fieldNorm.includes(queryNorm) || queryNorm.includes(fieldNorm)) score += 4.5;
        const tokenSet = new Set(tokenize(fieldNorm));
        queryTokens.forEach((token) => {
          if (token.length > 1 && tokenSet.has(token)) score += 1.15;
        });
      });
      if (normalize(claim.claim_text) === queryNorm) score += 6;
      return score + Number(claim.top_score || 0);
    }

    function scoreCorpus(query, item) {
      const queryNorm = normalize(query);
      if (!queryNorm) return 0;
      const itemNorm = normalize(item.text);
      if (!itemNorm) return 0;
      let score = 0;
      if (itemNorm.includes(queryNorm) || queryNorm.includes(itemNorm)) score += 3.4;
      const tokenSet = new Set(tokenize(itemNorm));
      tokenize(queryNorm).forEach((token) => {
        if (token.length > 1 && tokenSet.has(token)) score += 0.92;
      });
      return score;
    }

    function inferCorpusStatus(item) {
      if (!item) return "uncovered";
      const label = item.label;
      if (label === "真实") return "supported";
      if (label === "轻度夸大") return "partial";
      if (label === "因果颠倒") return "unsupported";
      if (label === "完全编造") return "unsupported";
      if (label === "语义鸿沟") return "uncovered";
      if (label === "否定性问题") return "uncovered";
      return "uncovered";
    }

    function corpusFallback(query) {
      const scored = payload.corpus
        .map((item) => ({ item, score: scoreCorpus(query, item) }))
        .sort((a, b) => b.score - a.score);
      if (!scored.length || scored[0].score < 2.2) return null;
      const picked = scored[0].item;
      const status = inferCorpusStatus(picked);
      const noteMap = {
        supported: "这句话暂时站得住。",
        partial: "它没全错，但说得太满。",
        unsupported: "它开始露出破绽了。",
        uncovered: "它滑出了当前图谱的光照范围。"
      };
      return {
        status,
        claim: null,
        score: scored[0].score,
        title: picked.text,
        note: noteMap[status],
        relation: `${picked.category} / ${picked.label_full}`,
        entities: "语料样例回退匹配",
        evidence: [
          `语料类别：${picked.category}`,
          `样例标签：${picked.label_full}`,
          ...payload.result_notes
        ]
      };
    }

    function resolveMatch(query) {
      const scored = payload.claims
        .map((claim) => ({ claim, score: scoreClaim(query, claim) }))
        .sort((a, b) => b.score - a.score);

      if (scored.length && scored[0].score >= 2.7) {
        const best = scored[0].claim;
        return {
          status: best.predicted_label === "supported" ? "supported" : "unsupported",
          claim: best,
          score: scored[0].score
        };
      }

      const fallback = corpusFallback(query);
      if (fallback) return fallback;

      return {
        status: "uncovered",
        claim: null,
        score: scored[0] ? scored[0].score : 0,
        title: query,
        note: "这句话暂时没有被抓住。",
        relation: "图谱未覆盖或样例不足",
        entities: "暂无明确实体对",
        evidence: payload.result_notes
      };
    }

    function renderResult(result, query) {
      const style = payload.styles[state.style];
      const top = qs("resultTop");
      const evidence = qs("evidenceList");
      const panel = qs("resultPanel");
      const revealAct3Btn = qs("revealAct3Btn");
      top.innerHTML = "";
      evidence.innerHTML = "";

      const badge = document.createElement("span");
      badge.className = `badge ${result.status}`;
      badge.textContent = (style.status_labels && style.status_labels[result.status]) || statusFallbacks[result.status] || "待判断";
      top.appendChild(badge);

      const standard = document.createElement("span");
      standard.className = "badge soft";
      standard.textContent = `标准判定：${verdictLabels[result.status] || result.status}`;
      top.appendChild(standard);

      panel.classList.add("is-revealed");
      panel.dataset.status = result.status;
      if (revealAct3Btn) revealAct3Btn.hidden = false;
      qs("verdictWord").textContent = badge.textContent;
      qs("evidenceHint").textContent = result.status === "supported" ? "它暂时站住了" : result.status === "unsupported" ? "裂缝已经出来了" : "这里只剩模糊边缘";

      if (result.claim) {
        const claim = result.claim;
        qs("resultTitle").textContent = claim.claim_text;
        qs("resultNote").textContent = claim.note;
        qs("resultRelation").textContent = claim.display_relation_label;
        qs("resultEntities").textContent = `${claim.x_name} → ${claim.y_name}`;
        qs("resultScore").textContent = `${result.score.toFixed(2)} / 检索 ${claim.top_score}`;

        ["relation_group_label", "false_type_label"].forEach((key) => {
          const meta = document.createElement("span");
          meta.className = "badge soft";
          meta.textContent = claim[key];
          top.appendChild(meta);
        });

        claim.evidence.forEach((item, index) => {
          const div = document.createElement("div");
          div.className = "evidence-item";
          div.style.animationDelay = `${index * 90}ms`;
          div.textContent = item;
          evidence.appendChild(div);
        });
        return;
      }

      qs("resultTitle").textContent = result.title || query || "当前没有输入";
      qs("resultNote").textContent = result.note || "暂无说明";
      qs("resultRelation").textContent = result.relation || "图谱未覆盖";
      qs("resultEntities").textContent = result.entities || "暂无明确实体对";
      qs("resultScore").textContent = result.score ? result.score.toFixed(2) : "--";

      (result.evidence || payload.result_notes).forEach((item, index) => {
        const div = document.createElement("div");
        div.className = "evidence-item";
        div.style.animationDelay = `${index * 90}ms`;
        div.textContent = item;
        evidence.appendChild(div);
      });
    }

    function runVerification(withLoading = true) {
      const query = qs("queryInput").value.trim() || payload.placeholders[0] || "";
      if (!query) return;

      const execute = () => {
        const result = resolveMatch(query);
        state.lastStatus = result.status;
        renderResult(result, query);
      };

      if (!withLoading) return execute();

      const shell = qs("scannerShell");
      const panel = qs("resultPanel");
      const rect = shell.getBoundingClientRect();
      shell.classList.add("is-active");
      panel.classList.add("is-active");
      document.body.classList.add("is-scanning");
      if (window.triggerFieldPulse) {
        window.triggerFieldPulse(rect.left + rect.width * 0.5, rect.top + rect.height * 0.5);
      }
      state.loadingIndex += 1;
      qs("loadingLine").textContent = payload.loading_texts[state.loadingIndex % Math.max(payload.loading_texts.length, 1)] || "正在图谱中检索…";
      window.setTimeout(() => {
        shell.classList.remove("is-active");
        document.body.classList.remove("is-scanning");
        execute();
      }, 960);
    }

    function renderSummaryTables() {
      const root = qs("summaryTables");
      root.innerHTML = "";
      payload.summary_tables.forEach((table) => {
        const card = document.createElement("article");
        card.className = "summary-card";
        card.innerHTML = `
          <h3>${table.title}</h3>
          <table>
            <tbody>${table.rows.map((row) => `<tr><td>${row.label}</td><td>${row.value}</td></tr>`).join("")}</tbody>
          </table>
        `;
        root.appendChild(card);
      });
    }

    function renderRiskCards() {
      const root = qs("riskCards");
      root.innerHTML = "";
      payload.panorama.risk_cards.forEach((card) => {
        const article = document.createElement("article");
        article.className = "risk-card";
        article.innerHTML = `
          <strong>${card.title}</strong>
          <p>${card.summary}</p>
          <ul>${card.examples.map((example) => `<li>${example}</li>`).join("")}</ul>
        `;
        root.appendChild(article);
      });
    }

    function renderPanorama() {
      const style = payload.styles[state.style];
      qs("panoramaIntro").textContent = style.panorama_intro || "查看聚合统计";
      qs("sliderLeft").textContent = style.slider_left || "纯模型回答";
      qs("sliderRight").textContent = style.slider_right || "图谱增强回答";
      qs("warmLine").textContent = style.warm_line || "";
      qs("closingLine").textContent = style.closing || "";

      const isRag = Number(state.slider) >= 100;
      qs("compareMode").textContent = isRag ? (style.slider_right || "图谱增强回答") : (style.slider_left || "纯模型回答");
      qs("compareAccuracy").textContent = formatPercent(isRag ? payload.panorama.rag_accuracy : payload.panorama.baseline_accuracy);
      qs("compareHallucination").textContent = formatPercent(isRag ? payload.panorama.rag_hallucination_rate : payload.panorama.baseline_hallucination_rate);
      qs("compareUnresolved").textContent = String(isRag ? payload.panorama.unresolved_count : Math.round(payload.claims.length * payload.panorama.baseline_hallucination_rate));
      renderSummaryTables();
      renderRiskCards();
    }

    function makeFilterChip(label, active, onClick) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-chip" + (active ? " is-active" : "");
      button.textContent = label;
      button.addEventListener("click", onClick);
      return button;
    }

    function renderCorpusFilters() {
      const categories = ["all", ...Array.from(new Set(payload.corpus.map((item) => item.category)))];
      const labels = ["all", ...Array.from(new Set(payload.corpus.map((item) => item.label)))];

      const catRoot = qs("categoryFilters");
      const labelRoot = qs("labelFilters");
      catRoot.innerHTML = "";
      labelRoot.innerHTML = "";

      categories.forEach((category) => {
        catRoot.appendChild(
          makeFilterChip(category === "all" ? "全部类别" : category, state.category === category, () => {
            state.category = category;
            renderCorpusFilters();
            renderCorpusLibrary();
          })
        );
      });

      labels.forEach((label) => {
        labelRoot.appendChild(
          makeFilterChip(label === "all" ? "全部标签" : label, state.label === label, () => {
            state.label = label;
            renderCorpusFilters();
            renderCorpusLibrary();
          })
        );
      });
    }

    function getFilteredCorpus() {
      return payload.corpus.filter((item) =>
        (state.category === "all" || item.category === state.category) &&
        (state.label === "all" || item.label === state.label)
      );
    }

    function renderCorpusLibrary() {
      const items = getFilteredCorpus();
      qs("corpusTotal").textContent = String(payload.corpus.length);
      qs("corpusFiltered").textContent = String(items.length);
      qs("corpusTopCategory").textContent = payload.corpus_stats.by_category[0]?.label || "--";
      qs("corpusTopLabel").textContent = payload.corpus_stats.by_label[0]?.label || "--";

      const root = qs("libraryList");
      root.innerHTML = "";
      items.forEach((item) => {
        const article = document.createElement("article");
        article.className = "library-item";
        article.innerHTML = `
          <div class="library-meta">
            <span class="badge soft">${item.category}</span>
            <span class="badge soft">${item.label_full}</span>
            <span class="badge soft">样例 ${item.id}</span>
          </div>
          <h4>${item.text}</h4>
          <div class="chip-row">
            <button class="ghost-btn accent" type="button">送去核查</button>
          </div>
        `;
        article.querySelector("button").addEventListener("click", () => {
          qs("queryInput").value = item.text;
          renderSuggestions();
          document.getElementById("verifyScene").scrollIntoView({ behavior: "smooth", block: "start" });
          window.setTimeout(() => runVerification(false), 220);
        });
        root.appendChild(article);
      });

      const notes = qs("usageNotes");
      notes.innerHTML = "";
      payload.usage_notes.forEach((text) => {
        const div = document.createElement("div");
        div.className = "usage-item";
        div.textContent = text;
        notes.appendChild(div);
      });
    }

    function pickRandomFrom(list) {
      if (!list.length) return null;
      return list[Math.floor(Math.random() * list.length)];
    }

    function fillAndVerify(text, withLoading = true) {
      showScene("verifyScene", false);
      qs("queryInput").value = text;
      renderSuggestions();
      document.getElementById("verifyScene").scrollIntoView({ behavior: "smooth", block: "start" });
      window.setTimeout(() => runVerification(withLoading), 220);
    }

    function showScene(sceneId, doScroll = true) {
      const section = document.getElementById(sceneId);
      if (!section) return;
      if (section.hidden) {
        section.hidden = false;
        requestAnimationFrame(() => section.classList.add("is-visible"));
      }
      if (doScroll) {
        section.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    function initFieldCanvas() {
      const canvas = qs("fieldCanvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      let width = 0;
      let height = 0;
      let nodes = [];
      let pulse = 0;
      let pulseX = 0;
      let pulseY = 0;

      function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        const count = Math.max(28, Math.min(64, Math.floor(width / 28)));
        nodes = Array.from({ length: count }, () => ({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.22,
          vy: (Math.random() - 0.5) * 0.22,
          r: Math.random() * 1.8 + 0.6,
        }));
      }

      function draw(time) {
        ctx.clearRect(0, 0, width, height);
        const warm = ctx.createRadialGradient(width * 0.18, height * 0.2, 0, width * 0.18, height * 0.2, width * 0.34);
        warm.addColorStop(0, "rgba(255, 133, 90, 0.08)");
        warm.addColorStop(1, "rgba(255, 133, 90, 0)");
        ctx.fillStyle = warm;
        ctx.fillRect(0, 0, width, height);

        const cold = ctx.createRadialGradient(width * 0.82, height * 0.18, 0, width * 0.82, height * 0.18, width * 0.3);
        cold.addColorStop(0, "rgba(141, 220, 255, 0.08)");
        cold.addColorStop(1, "rgba(141, 220, 255, 0)");
        ctx.fillStyle = cold;
        ctx.fillRect(0, 0, width, height);

        nodes.forEach((node, index) => {
          node.x += node.vx + Math.sin((time * 0.0004) + index) * 0.06;
          node.y += node.vy + Math.cos((time * 0.0003) + index) * 0.06;

          if (node.x < -20) node.x = width + 20;
          if (node.x > width + 20) node.x = -20;
          if (node.y < -20) node.y = height + 20;
          if (node.y > height + 20) node.y = -20;
        });

        for (let i = 0; i < nodes.length; i += 1) {
          for (let j = i + 1; j < nodes.length; j += 1) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 140) continue;
            ctx.strokeStyle = `rgba(141,220,255,${0.08 * (1 - dist / 140)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }

        nodes.forEach((node, index) => {
          const hue = index % 3 === 0 ? "255, 140, 96" : "141, 220, 255";
          ctx.fillStyle = `rgba(${hue},0.55)`;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
          ctx.fill();
        });

        if (pulse > 0) {
          const radius = 80 + (1 - pulse) * 520;
          ctx.strokeStyle = `rgba(255, 212, 134, ${pulse * 0.36})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(pulseX, pulseY, radius, 0, Math.PI * 2);
          ctx.stroke();
          pulse *= 0.956;
        }

        requestAnimationFrame(draw);
      }

      window.addEventListener("resize", resize);
      window.triggerFieldPulse = (x, y) => {
        pulse = 1;
        pulseX = x;
        pulseY = y;
      };

      resize();
      requestAnimationFrame(draw);
    }

    qs("queryInput").addEventListener("input", renderSuggestions);
    qs("queryInput").addEventListener("focus", () => {
      qs("scannerShell").classList.add("is-focus");
      renderSuggestions();
    });
    qs("queryInput").addEventListener("blur", () => {
      window.setTimeout(() => qs("scannerShell").classList.remove("is-focus"), 120);
    });
    qs("verifyButton").addEventListener("click", () => runVerification(true));
    qs("clearInputBtn").addEventListener("click", () => {
      qs("queryInput").value = "";
      state.loadingIndex += 1;
      renderVerifierCopy();
      renderSuggestions();
    });
    qs("compareSlider").addEventListener("input", (event) => {
      state.slider = Number(event.target.value);
      renderPanorama();
    });
    qs("enterAct2Btn").addEventListener("click", () => {
      showScene("verifyScene", true);
    });
    qs("revealAct3Btn").addEventListener("click", () => {
      showScene("panoramaScene", true);
    });
    qs("randomFalseBtn").addEventListener("click", () => {
      const picked = pickRandomFrom(payload.false_corpus_examples) || pickRandomFrom(payload.corpus);
      if (picked) fillAndVerify(picked.text, true);
    });
    qs("sampleMythBtn").addEventListener("click", () => {
      const picked = pickRandomFrom(payload.false_corpus_examples) || payload.corpus[0];
      if (picked) fillAndVerify(picked.text, true);
    });

    document.addEventListener("mousemove", (event) => {
      const x = `${(event.clientX / window.innerWidth) * 100}%`;
      const y = `${(event.clientY / window.innerHeight) * 100}%`;
      document.documentElement.style.setProperty("--mouse-x", x);
      document.documentElement.style.setProperty("--mouse-y", y);
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    }, { threshold: 0.16 });

    document.querySelectorAll(".scene").forEach((section, index) => {
      if (index === 0) {
        section.classList.add("is-visible");
      }
      observer.observe(section);
    });

    applyStyleTheme();
    initFieldCanvas();
    renderHero();
    initHeroParallax();
    renderVerifierCopy();
    renderPanorama();
    qs("queryInput").value = payload.placeholders[0] || "";
    renderSuggestions();
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", payload["meta"]["title"])
        .replace("__GENERATED_AT__", payload["meta"]["generated_at"])
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def write_demo_files(project_root: Path, payload: dict) -> tuple[Path, Path]:
    demo_root = ensure_demo_directories(project_root)["demo"]
    payload_path = demo_root / "demo_payload.json"
    html_path = demo_root / "index.html"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_demo_html(payload), encoding="utf-8")
    return html_path, payload_path


def write_demo_summary(project_root: Path, payload: dict, html_path: Path, payload_path: Path) -> Path:
    summary = {
        "html_path": str(html_path),
        "payload_path": str(payload_path),
        "recommended_url": "http://127.0.0.1:8765/",
        "claim_count": len(payload["claims"]),
        "corpus_count": len(payload["corpus"]),
        "generated_at": payload["meta"]["generated_at"],
        "note": "Open demo/index.html directly, or visit the local demo server if it is running.",
    }
    summary_path = project_root / "data" / "reports" / "step6_demo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
