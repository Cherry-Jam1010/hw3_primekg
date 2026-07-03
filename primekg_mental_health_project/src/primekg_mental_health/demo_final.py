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
    "No exact triplet exists in current PrimeKG subgraph.": "当前 PrimeKG 子图中不存在该精确三元组。",
    "Retriever confidence is too low.": "检索置信度过低。",
    "Relevant evidence not retrieved in top-k.": "Top-k 检索未命中相关证据。",
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


def translate_note(note: str) -> str:
    return NOTE_TRANSLATIONS.get(note, note)


def translate_relation_label(relation: str) -> str:
    return DISPLAY_RELATION_LABELS.get(relation, relation)


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
        return f"{disease_name} 的疾病摘要。"

    for english, chinese in replacements:
        if english not in text:
            continue
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
    reports_root = project_root / "data" / "reports"
    demo_root.mkdir(parents=True, exist_ok=True)
    return {"demo": demo_root, "reports": reports_root}


def extract_section(text: str, start_marker: str, end_markers: list[str]) -> str:
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    tail = text[start_idx:]
    positions = [tail.find(marker) for marker in end_markers if tail.find(marker) != -1]
    if not positions:
        return tail.strip()
    return tail[: min(positions)].strip()


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


def extract_table_after_marker(text: str, marker: str) -> list[dict[str, str]]:
    pattern = re.compile(rf".*{re.escape(marker)}.*\n((?:\|.*(?:\n|$))+)", re.MULTILINE)
    match = pattern.search(text)
    return parse_markdown_table(match.group(1)) if match else []


def extract_bullets_after_marker(text: str, marker: str) -> list[str]:
    pattern = re.compile(rf".*{re.escape(marker)}.*\n((?:- .+(?:\n|$))+)", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return []
    return [line[2:].strip() for line in match.group(1).splitlines() if line.strip().startswith("- ")]


def parse_version_bullets(section_text: str) -> dict[str, list[str]]:
    versions = {style_key: [] for style_key in STYLE_LABELS}
    pattern = re.compile(r"\*\*(版本[ABC]\s*·\s*(克制版|冲击版|学术版))\*\*\s*\n((?:- .+(?:\n|$))+)")
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
        if "版本A" in prefix:
            output["restrained"] = value.strip()
        elif "版本B" in prefix:
            output["impact"] = value.strip()
        elif "版本C" in prefix:
            output["academic"] = value.strip()
    return output


def parse_plan_document(plan_path: Path) -> dict:
    text = plan_path.read_text(encoding="utf-8")
    scene1 = extract_section(text, "### 第一幕：茧房态（氛围文案，可选，浮动/半透明展示）", ["---", "### 第二幕"])
    scene2 = extract_section(text, "### 第二幕：核查交互区", ["---", "### 第三幕"])
    scene3 = extract_section(text, "### 第三幕：数据全景区", ["---", "## 三、测试语料库"])
    corpus_section = extract_section(text, "## 三、测试语料库（约50条，含真假混合）", ["---", "## 四、使用建议"])
    usage_section = extract_section(text, "## 四、使用建议", [])

    styles: dict[str, dict] = {style_key: {"label": label} for style_key, label in STYLE_LABELS.items()}
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
        styles[style_key]["status_labels"] = {}

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

    usage_notes = [line[2:].strip() for line in usage_section.splitlines() if line.strip().startswith("- ")]
    return {
        "styles": styles,
        "placeholders": extract_bullets_after_marker(scene2, "输入框placeholder"),
        "loading_texts": extract_bullets_after_marker(scene2, "加载态文案"),
        "result_notes": extract_bullets_after_marker(scene2, "结果附加说明"),
        "usage_notes": usage_notes,
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
                "claim_text": str(series["claim_text"]).strip(),
                "relation_group": str(series["relation_group"]),
                "relation_group_label": RELATION_GROUP_LABELS.get(str(series["relation_group"]), str(series["relation_group"])),
                "false_type": str(series["false_type"]),
                "false_type_label": FALSE_TYPE_LABELS.get(str(series["false_type"]), str(series["false_type"])),
                "predicted_label": predicted_label,
                "predicted_label_text": VERDICT_LABELS.get(predicted_label, predicted_label),
                "top_score": round(float(series.get("top_score", 0.0)), 3),
                "x_name": str(series["x_name"]),
                "y_name": str(series["y_name"]),
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
            "summary": "需要跨越多条关系才能判断。",
            "examples": [corpus_map[item_id]["text"] for item_id in (44, 45) if item_id in corpus_map],
        },
        {
            "title": "语义鸿沟",
            "summary": "口语热词与图谱术语不一定直接对齐。",
            "examples": [corpus_map[item_id]["text"] for item_id in (46, 47) if item_id in corpus_map],
        },
        {
            "title": "否定性问题",
            "summary": "“不存在任何关系”这类说法更难被普通检索解释清楚。",
            "examples": [corpus_map[item_id]["text"] for item_id in (48, 49) if item_id in corpus_map],
        },
        {
            "title": "虚构实体",
            "summary": "迎合式幻觉会把不存在的诊断讲得像真的一样。",
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
    return [{"name": row["node_name"], "degree": int(row["degree"])} for row in step1_summary.get("top_disease_nodes", [])[:limit]]


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

    all_claims = build_claim_records(rag_predictions)
    featured_claims = pick_featured_claims(all_claims, sample_limit)
    pure_row = rag_summary_df.loc[rag_summary_df["system"] == "pure_llm_simulated"].iloc[0]
    rag_row = rag_summary_df.loc[rag_summary_df["system"] == "rag_verifier"].iloc[0]

    return {
        "meta": {
            "title": "PrimeKG 精神健康知识图谱最终演示方案",
            "subtitle": "三幕式叙事、图谱核查交互与测试语料库一体化前端",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "styles": plan_data["styles"],
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
      --bg: #f7f2ea;
      --panel: rgba(255, 251, 246, 0.82);
      --panel-strong: rgba(255, 252, 248, 0.94);
      --line: rgba(27, 36, 41, 0.08);
      --text: #19262d;
      --muted: #66767c;
      --teal: #0f766e;
      --amber: #c48a18;
      --coral: #c75c3d;
      --ink: #244854;
      --shadow: 0 22px 64px rgba(27, 36, 41, 0.1);
      --radius-xl: 30px;
      --radius-lg: 24px;
      --sans: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
      --serif: "STSong", "SimSun", "Source Han Serif SC", serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 8% 0%, rgba(15,118,110,0.16), transparent 26%),
        radial-gradient(circle at 92% 10%, rgba(196,138,24,0.16), transparent 24%),
        linear-gradient(180deg, #fbf8f3 0%, var(--bg) 100%);
    }
    .page { width: min(1360px, calc(100vw - 32px)); margin: 0 auto; padding: 18px 0 48px; }
    .topbar {
      position: sticky; top: 12px; z-index: 30; display: flex; align-items: center; justify-content: space-between;
      gap: 14px; margin-bottom: 18px; padding: 14px 18px; border-radius: 999px;
      background: rgba(255, 252, 248, 0.86); border: 1px solid rgba(255,255,255,0.72);
      box-shadow: var(--shadow); backdrop-filter: blur(10px);
    }
    .brand { display: flex; flex-direction: column; gap: 4px; }
    .brand strong { font-size: 0.98rem; letter-spacing: 0.04em; }
    .brand span { color: var(--muted); font-size: 0.84rem; }
    .switcher { display: flex; flex-wrap: wrap; gap: 8px; }
    .pill, .quick-chip, .status-chip, .filter-chip, .ghost-btn {
      border: 0; border-radius: 999px; padding: 9px 14px; background: rgba(25,38,45,0.07);
      color: var(--text); cursor: pointer; font-weight: 700; font: inherit;
    }
    .pill.is-active, .filter-chip.is-active { background: var(--text); color: #fff; }
    .act {
      margin-top: 18px; border-radius: var(--radius-xl); border: 1px solid rgba(255,255,255,0.72);
      background: var(--panel); box-shadow: var(--shadow); backdrop-filter: blur(10px); overflow: hidden;
    }
    .act-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 22px 0; }
    .act-head h2 { margin: 0; font-size: 1rem; letter-spacing: 0.12em; color: var(--teal); }
    .act-head span { color: var(--muted); font-size: 0.84rem; }
    .hero, .verifier, .insight-grid { display: grid; gap: 18px; }
    .hero { grid-template-columns: 1.15fr 0.85fr; padding: 18px 22px 26px; }
    .hero-main {
      position: relative; min-height: 380px; padding: 28px; border-radius: var(--radius-lg); overflow: hidden;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,241,230,0.96));
    }
    .hero-main::after {
      content: ""; position: absolute; right: -36px; bottom: -42px; width: 220px; height: 220px;
      border-radius: 42px; transform: rotate(16deg);
      background: linear-gradient(145deg, rgba(15,118,110,0.12), rgba(196,138,24,0.14));
    }
    .eyebrow {
      display: inline-flex; padding: 8px 14px; border-radius: 999px; background: rgba(15,118,110,0.1);
      color: var(--teal); font-weight: 700; font-size: 0.84rem;
    }
    .hero h1 {
      position: relative; z-index: 1; margin: 18px 0 10px; font-family: var(--serif);
      font-size: clamp(2.35rem, 4vw, 3.7rem); line-height: 1.05; letter-spacing: -0.04em;
    }
    .hero-subtitle { position: relative; z-index: 1; margin: 0; max-width: 36rem; color: var(--muted); line-height: 1.75; font-size: 0.98rem; }
    .hero-stats, .cocoon-wall, .summary-grid, .compare-metrics, .library-stats, .risk-grid, .featured-grid, .result-grid {
      display: grid; gap: 12px;
    }
    .hero-stats { position: relative; z-index: 1; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 22px; }
    .hero-stat, .summary-card, .compare-metric, .mini-stat, .library-item, .featured-card, .result-box {
      padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.84); border: 1px solid var(--line);
    }
    .hero-stat span, .compare-metric span, .mini-stat span, .result-box span { display: block; color: var(--muted); font-size: 0.8rem; margin-bottom: 8px; }
    .hero-stat strong, .compare-metric strong, .mini-stat strong { font-family: var(--serif); font-size: 1.24rem; }
    .cocoon-wall { grid-template-columns: repeat(2, minmax(0, 1fr)); align-content: start; }
    .floating-card {
      position: relative; overflow: hidden; min-height: 112px; padding: 18px; border-radius: 22px;
      background: linear-gradient(160deg, rgba(255,255,255,0.82), rgba(247,240,229,0.92)); border: 1px solid var(--line);
    }
    .floating-card::after {
      content: ""; position: absolute; right: -22px; top: -22px; width: 92px; height: 92px;
      border-radius: 26px; transform: rotate(20deg); background: rgba(196,138,24,0.08);
    }
    .floating-card small { position: relative; z-index: 1; display: block; color: var(--teal); font-weight: 700; font-size: 0.78rem; margin-bottom: 8px; }
    .floating-card p { position: relative; z-index: 1; margin: 0; font-size: 1rem; line-height: 1.55; }
    .verifier { grid-template-columns: 1.02fr 0.98fr; padding: 18px 22px 24px; }
    .panel, .compare-box, .bar-panel, .risk-panel, .library-panel {
      border-radius: var(--radius-lg); background: var(--panel-strong); border: 1px solid var(--line); padding: 18px;
    }
    .panel h3, .bar-panel h3, .risk-panel h3, .library-panel h3 { margin: 0 0 8px; font-size: 1.1rem; }
    .panel-caption { color: var(--muted); font-size: 0.92rem; line-height: 1.65; margin-bottom: 14px; }
    .input-shell, .result-shell, .panorama, .library-wrap, .footer-notes, .bar-stack, .node-list, .risk-grid, .library-list { display: grid; gap: 12px; }
    .query-box { position: relative; }
    .query-input {
      width: 100%; min-height: 132px; resize: vertical; border: 1px solid rgba(25,38,45,0.1); border-radius: 22px;
      padding: 16px 18px; font: inherit; line-height: 1.7; background: rgba(255,255,255,0.9); color: var(--text); outline: none;
    }
    .suggestions {
      position: absolute; top: calc(100% + 8px); left: 0; right: 0; display: none; z-index: 12; padding: 10px;
      border-radius: 18px; background: rgba(255,255,255,0.98); border: 1px solid var(--line); box-shadow: 0 18px 40px rgba(25,38,45,0.12);
    }
    .suggestions.is-open { display: block; }
    .suggestion-list { display: grid; gap: 8px; max-height: 280px; overflow: auto; }
    .suggestion-item {
      display: flex; justify-content: space-between; gap: 10px; width: 100%; border: 0; border-radius: 14px; padding: 10px 12px;
      background: rgba(244,239,231,0.78); color: var(--text); text-align: left; cursor: pointer;
    }
    .suggestion-item span:last-child { color: var(--muted); font-size: 0.8rem; white-space: nowrap; }
    .quick-row, .tag-row, .cta-row, .result-top, .meta, .actions, .library-filters, .library-top { display: flex; flex-wrap: wrap; gap: 10px; }
    .primary-btn {
      border: 0; border-radius: 18px; padding: 14px 20px; background: linear-gradient(135deg, var(--text), #244854);
      color: #fff; font: inherit; font-weight: 700; cursor: pointer; min-width: 136px;
    }
    .primary-btn.is-loading { opacity: 0.82; pointer-events: none; }
    .loading-line { color: var(--muted); font-size: 0.9rem; }
    .badge {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 12px; font-size: 0.82rem; font-weight: 700;
    }
    .badge.supported { background: rgba(15,118,110,0.12); color: var(--teal); }
    .badge.partial { background: rgba(196,138,24,0.16); color: #97660d; }
    .badge.unsupported { background: rgba(199,92,61,0.12); color: var(--coral); }
    .badge.uncovered, .badge.soft { background: rgba(25,38,45,0.08); color: var(--muted); }
    .result-title { margin: 0; font-size: 1.2rem; line-height: 1.6; }
    .result-note { padding: 14px 16px; border-radius: 18px; background: rgba(15,118,110,0.08); color: #244854; line-height: 1.7; font-size: 0.94rem; }
    .result-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .evidence-list { display: grid; gap: 10px; }
    .evidence-item { padding: 12px 14px; border-radius: 16px; background: rgba(255,255,255,0.86); border-left: 4px solid var(--amber); line-height: 1.65; font-size: 0.92rem; }
    .featured-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .compare-head, .slider-labels, .bar-label, .node-row { display: flex; justify-content: space-between; gap: 12px; }
    .compare-head strong { font-size: 1.08rem; }
    .compare-head span, .slider-labels, .bar-label, .node-row span, .footer-note { color: var(--muted); }
    .compare-slider { width: 100%; accent-color: var(--teal); }
    .compare-metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 16px; }
    .summary-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .summary-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .summary-table tr + tr { border-top: 1px solid rgba(25,38,45,0.08); }
    .summary-table td { padding: 10px 0; vertical-align: top; }
    .summary-table td:first-child { color: var(--muted); width: 44%; padding-right: 10px; }
    .summary-table td:last-child { text-align: right; font-weight: 700; }
    .insight-grid { grid-template-columns: 0.94fr 1.06fr; }
    .bar-row { display: grid; gap: 6px; }
    .bar-track { height: 12px; border-radius: 999px; background: rgba(25,38,45,0.08); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--teal), var(--amber)); }
    .node-row { padding: 12px 14px; border-radius: 16px; background: rgba(248,243,236,0.9); border: 1px solid rgba(25,38,45,0.06); }
    .node-row strong { max-width: 80%; font-size: 0.92rem; line-height: 1.5; }
    .risk-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .risk-card { padding: 16px; border-radius: 20px; background: linear-gradient(180deg, rgba(249,239,226,0.92), rgba(255,250,244,0.92)); border: 1px solid rgba(196,138,24,0.14); }
    .risk-card strong { display: block; margin-bottom: 8px; font-size: 1rem; }
    .risk-card p { margin: 0 0 12px; color: var(--muted); line-height: 1.6; font-size: 0.9rem; }
    .risk-card ul { margin: 0; padding-left: 18px; color: var(--text); line-height: 1.65; font-size: 0.88rem; }
    .closing-line { padding: 14px 18px; border-radius: 20px; background: rgba(25,38,45,0.94); color: #fff; text-align: center; font-size: 0.96rem; letter-spacing: 0.02em; }
    .library-wrap { padding: 18px 22px 24px; }
    .library-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); min-width: 320px; }
    .library-item h4 { margin: 10px 0 0; font-size: 1rem; line-height: 1.65; }
    .ghost-btn { border: 1px solid rgba(25,38,45,0.1); background: rgba(248,243,236,0.9); }
    .footer-note { padding: 12px 14px; border-radius: 16px; background: rgba(255,255,255,0.82); border: 1px solid var(--line); line-height: 1.65; font-size: 0.9rem; }
    @media (max-width: 1120px) {
      .hero, .verifier, .insight-grid, .library-top, .summary-grid, .risk-grid, .featured-grid { grid-template-columns: 1fr; }
      .hero-stats, .compare-metrics, .library-stats, .result-grid, .cocoon-wall { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .page { width: min(100vw - 18px, 1360px); padding-top: 12px; }
      .topbar { border-radius: 28px; align-items: flex-start; }
      .hero-stats, .compare-metrics, .library-stats, .result-grid, .cocoon-wall { grid-template-columns: 1fr; }
      .act-head { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <strong>PrimeKG 精神健康知识图谱最终演示</strong>
        <span>根据 `plan.md` 的三幕式前端方案生成，构建时间 __GENERATED_AT__</span>
      </div>
      <div class="switcher" id="styleSwitcher"></div>
    </header>
    <section class="act">
      <div class="act-head"><h2>ACT 01</h2><span>茧房态 / 半透明叙事前幕</span></div>
      <div class="hero">
        <div class="hero-main">
          <div class="eyebrow">PrimeKG × 精神健康声明核查</div>
          <h1>看起来像真相的说法，先别急着信。</h1>
          <p class="hero-subtitle" id="heroSubtitle"></p>
          <div class="hero-stats">
            <div class="hero-stat"><span>种子疾病</span><strong id="seedCount"></strong></div>
            <div class="hero-stat"><span>实验声明</span><strong id="claimCount"></strong></div>
            <div class="hero-stat"><span>RAG 校验准确率</span><strong id="ragAcc"></strong></div>
          </div>
        </div>
        <div class="cocoon-wall" id="cocoonWall"></div>
      </div>
    </section>
    <section class="act">
      <div class="act-head"><h2>ACT 02</h2><span>核查交互区 / 输入一句说法，立刻回看图谱证据</span></div>
      <div class="verifier">
        <div class="panel">
          <h3 id="entryPrompt"></h3>
          <div class="panel-caption">可以直接粘贴一句刷到的说法，也可以点击下方样例快速填入。</div>
          <div class="input-shell">
            <div class="query-box">
              <textarea id="queryInput" class="query-input" placeholder=""></textarea>
              <div id="suggestions" class="suggestions"><div id="suggestionList" class="suggestion-list"></div></div>
            </div>
            <div class="quick-row" id="placeholderChips"></div>
            <div class="cta-row"><button id="verifyButton" class="primary-btn"></button><span id="loadingLine" class="loading-line"></span></div>
            <div class="tag-row" id="featuredClaims"></div>
          </div>
        </div>
        <div class="panel">
          <h3>核查结果</h3>
          <div class="panel-caption">优先返回与输入最接近的实验样例；如果当前实验集里没有足够接近的证据，会明确标注为“无相关记录”。</div>
          <div class="result-shell">
            <div class="result-top" id="resultTop"></div>
            <h3 class="result-title" id="resultTitle">等待输入</h3>
            <div class="result-note" id="resultNote">这里会展示判定说明、匹配到的关系类型，以及当前可见的图谱证据。</div>
            <div class="result-grid">
              <div class="result-box"><span>实体对</span><strong id="resultEntities">--</strong></div>
              <div class="result-box"><span>关系</span><strong id="resultRelation">--</strong></div>
              <div class="result-box"><span>匹配分数</span><strong id="resultScore">--</strong></div>
            </div>
            <div class="evidence-list" id="evidenceList"></div>
          </div>
        </div>
      </div>
    </section>
    <section class="act">
      <div class="act-head"><h2>ACT 03</h2><span>数据全景区 / 从纯模型到图谱增强</span></div>
      <div class="panorama">
        <div class="compare-box">
          <div class="compare-head"><strong id="panoramaIntro"></strong><span id="warmLine"></span></div>
          <div class="slider-labels"><span id="sliderLeft"></span><span id="sliderRight"></span></div>
          <input id="compareSlider" class="compare-slider" type="range" min="0" max="100" step="100" value="0">
          <div class="compare-metrics">
            <div class="compare-metric"><span>当前模式</span><strong id="compareMode"></strong></div>
            <div class="compare-metric"><span>准确率</span><strong id="compareAccuracy"></strong></div>
            <div class="compare-metric"><span>幻觉率</span><strong id="compareHallucination"></strong></div>
          </div>
        </div>
        <div class="summary-grid" id="summaryTables"></div>
        <div class="insight-grid">
          <div class="bar-panel">
            <h3>精神障碍家族分布</h3>
            <div class="panel-caption">当前子图主要聚焦在可解释、可展示的精神健康相关疾病家族。</div>
            <div id="familyBars" class="bar-stack"></div>
            <h3 style="margin-top:16px;">高连接疾病节点</h3>
            <div id="topNodes" class="node-list"></div>
          </div>
          <div class="risk-panel">
            <h3>边界与风险卡</h3>
            <div class="panel-caption">这些不是为了夸大，而是为了把最值得在汇报里说明的高风险场景留在页面上。</div>
            <div id="riskCards" class="risk-grid"></div>
          </div>
        </div>
        <div id="closingLine" class="closing-line"></div>
      </div>
    </section>
    <section class="act">
      <div class="act-head"><h2>APPENDIX</h2><span>测试语料库 / 来自 `plan.md` 的 50 条前端样例</span></div>
      <div class="library-wrap">
        <div class="library-top">
          <div class="library-panel" style="flex:1 1 620px;">
            <h3>测试语料库</h3>
            <div class="panel-caption">这些语料是前端设计方案里的展示样例，用于补全“刷到一句说法就想核查”的真实使用场景；其中部分并未与当前 PrimeKG 子图逐条对齐，正式提交前仍需按实验数据回填。</div>
            <div class="library-filters">
              <div class="tag-row" id="categoryFilters"></div>
              <div class="tag-row" id="labelFilters"></div>
            </div>
          </div>
          <div class="library-stats">
            <div class="mini-stat"><span>语料总数</span><strong id="corpusTotal"></strong></div>
            <div class="mini-stat"><span>当前筛选结果</span><strong id="corpusFiltered"></strong></div>
            <div class="mini-stat"><span>主要类别</span><strong id="corpusTopCategory"></strong></div>
            <div class="mini-stat"><span>主要标签</span><strong id="corpusTopLabel"></strong></div>
          </div>
        </div>
        <div id="libraryList" class="library-list"></div>
        <div class="footer-notes" id="usageNotes"></div>
      </div>
    </section>
  </div>
  <script>
    const payload = __PAYLOAD_JSON__;
    const state = { style: "academic", slider: 0, category: "all", label: "all", loadingIndex: 0 };
    const qs = (id) => document.getElementById(id);
    const normalize = (text) => String(text || "").toLowerCase().replace(/[^\u4e00-\u9fff\w]+/g, " ").replace(/\s+/g, " ").trim();
    const tokenize = (text) => normalize(text).match(/[\u4e00-\u9fff]+|[a-z0-9_-]+/g) || [];
    const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

    function makeFilterChip(label, active, onClick) {
      const button = document.createElement("button");
      button.className = "filter-chip" + (active ? " is-active" : "");
      button.textContent = label;
      button.addEventListener("click", onClick);
      return button;
    }

    function renderStyleSwitcher() {
      const root = qs("styleSwitcher");
      root.innerHTML = "";
      Object.entries(payload.styles).forEach(([styleKey, info]) => {
        const button = document.createElement("button");
        button.className = "pill" + (state.style === styleKey ? " is-active" : "");
        button.textContent = info.label;
        button.addEventListener("click", () => {
          state.style = styleKey;
          renderStyleSwitcher();
          renderActOne();
          renderActTwoText();
          renderPanorama();
        });
        root.appendChild(button);
      });
    }

    function renderActOne() {
      const style = payload.styles[state.style];
      qs("heroSubtitle").textContent = payload.meta.subtitle;
      qs("seedCount").textContent = payload.summary_tables[0].rows[0].value;
      qs("claimCount").textContent = String(payload.claims.length);
      qs("ragAcc").textContent = formatPercent(payload.panorama.rag_accuracy);
      const wall = qs("cocoonWall");
      wall.innerHTML = "";
      style.floating_lines.forEach((line, index) => {
        const card = document.createElement("article");
        card.className = "floating-card";
        card.innerHTML = `<small>片段 ${String(index + 1).padStart(2, "0")}</small><p>${line}</p>`;
        wall.appendChild(card);
      });
    }

    function renderActTwoText() {
      const style = payload.styles[state.style];
      qs("entryPrompt").textContent = style.entry_prompt || "输入待核验陈述";
      qs("verifyButton").textContent = style.button_label || "提交核验";
      qs("loadingLine").textContent = payload.loading_texts[state.loadingIndex % payload.loading_texts.length] || "";
      const input = qs("queryInput");
      if (!input.value.trim()) input.placeholder = payload.placeholders[state.loadingIndex % payload.placeholders.length] || "";

      const chips = qs("placeholderChips");
      chips.innerHTML = "";
      payload.placeholders.forEach((text) => {
        const button = document.createElement("button");
        button.className = "quick-chip";
        button.textContent = text;
        button.addEventListener("click", () => { input.value = text; closeSuggestions(); runVerification(false); });
        chips.appendChild(button);
      });

      const featured = qs("featuredClaims");
      featured.innerHTML = "";
      payload.featured_claims.forEach((claim) => {
        const button = document.createElement("button");
        button.className = "status-chip";
        button.textContent = claim.claim_text.length > 24 ? claim.claim_text.slice(0, 24) + "…" : claim.claim_text;
        button.addEventListener("click", () => { input.value = claim.claim_text; closeSuggestions(); runVerification(false); });
        featured.appendChild(button);
      });
    }

    function buildSuggestionItems(query) {
      const queryNorm = normalize(query);
      const items = [];
      payload.featured_claims.forEach((claim) => {
        if (!queryNorm || normalize(claim.claim_text).includes(queryNorm) || normalize(claim.x_name).includes(queryNorm) || normalize(claim.y_name).includes(queryNorm)) {
          items.push({ label: claim.claim_text, hint: "实验样例", value: claim.claim_text });
        }
      });
      payload.corpus.forEach((item) => {
        if (!queryNorm || normalize(item.text).includes(queryNorm)) {
          items.push({ label: item.text, hint: `${item.category} / ${item.label}`, value: item.text });
        }
      });
      return items.slice(0, 10);
    }

    function renderSuggestions() {
      const root = qs("suggestionList");
      root.innerHTML = "";
      const items = buildSuggestionItems(qs("queryInput").value);
      if (!items.length) return closeSuggestions();
      items.forEach((item) => {
        const button = document.createElement("button");
        button.className = "suggestion-item";
        button.type = "button";
        button.innerHTML = `<span>${item.label}</span><span>${item.hint}</span>`;
        button.addEventListener("click", () => { qs("queryInput").value = item.value; closeSuggestions(); });
        root.appendChild(button);
      });
      qs("suggestions").classList.add("is-open");
    }

    function closeSuggestions() { qs("suggestions").classList.remove("is-open"); }

    function scoreClaim(query, claim) {
      const queryNorm = normalize(query);
      if (!queryNorm) return 0;
      const queryTokens = tokenize(queryNorm);
      const fields = [claim.claim_text, claim.x_name, claim.y_name, claim.display_relation_label, claim.relation_group_label, claim.false_type_label];
      let score = 0;
      fields.forEach((field) => {
        const fieldNorm = normalize(field);
        if (!fieldNorm) return;
        if (fieldNorm.includes(queryNorm) || queryNorm.includes(fieldNorm)) score += 4.4;
        const tokenSet = new Set(tokenize(fieldNorm));
        queryTokens.forEach((token) => { if (token.length > 1 && tokenSet.has(token)) score += 1.15; });
      });
      if (normalize(claim.claim_text) === queryNorm) score += 6;
      return score + Number(claim.top_score || 0);
    }

    function resolveMatch(query) {
      const scored = payload.claims.map((claim) => ({ claim, score: scoreClaim(query, claim) })).sort((a, b) => b.score - a.score);
      if (!scored.length) return { status: "uncovered", claim: null, score: 0 };
      const best = scored[0];
      if (best.score < 2.6) return { status: "uncovered", claim: best.claim, score: best.score };
      return { status: best.claim.predicted_label === "supported" ? "supported" : "unsupported", claim: best.claim, score: best.score };
    }

    function renderResult(result, query) {
      const style = payload.styles[state.style];
      const top = qs("resultTop");
      const evidence = qs("evidenceList");
      top.innerHTML = "";
      evidence.innerHTML = "";

      const badge = document.createElement("span");
      badge.className = `badge ${result.status}`;
      badge.textContent = style.status_labels[result.status] || VERDICT_LABELS[result.status] || "待判断";
      top.appendChild(badge);

      const standard = document.createElement("span");
      standard.className = "badge soft";
      standard.textContent = result.status === "uncovered" ? "图谱未覆盖" : (result.status === "supported" ? "标准判定：支持" : "标准判定：不支持");
      top.appendChild(standard);

      if (!result.claim || result.status === "uncovered") {
        qs("resultTitle").textContent = query || "当前没有输入";
        qs("resultNote").textContent = "当前实验样例中没有找到足够接近、可直接回溯到图谱证据的说法。可以改用下方语料库样例，或换一个更具体的表述再试一次。";
        qs("resultEntities").textContent = "暂无明确实体对";
        qs("resultRelation").textContent = "图谱未覆盖或样例不足";
        qs("resultScore").textContent = result.score ? result.score.toFixed(2) : "--";
        payload.result_notes.forEach((item) => {
          const div = document.createElement("div");
          div.className = "evidence-item";
          div.textContent = item;
          evidence.appendChild(div);
        });
        return;
      }

      const claim = result.claim;
      qs("resultTitle").textContent = claim.claim_text;
      qs("resultNote").textContent = claim.note;
      qs("resultEntities").textContent = `${claim.x_name} → ${claim.y_name}`;
      qs("resultRelation").textContent = claim.display_relation_label;
      qs("resultScore").textContent = `${result.score.toFixed(2)} / 检索 ${claim.top_score}`;

      ["relation_group_label", "false_type_label"].forEach((key) => {
        const meta = document.createElement("span");
        meta.className = "badge soft";
        meta.textContent = claim[key];
        top.appendChild(meta);
      });

      claim.evidence.forEach((item) => {
        const div = document.createElement("div");
        div.className = "evidence-item";
        div.textContent = item;
        evidence.appendChild(div);
      });
    }

    function runVerification(withLoading = true) {
      const query = qs("queryInput").value.trim() || payload.placeholders[0] || "";
      if (!query) return;
      const execute = () => renderResult(resolveMatch(query), query);
      if (!withLoading) return execute();
      const button = qs("verifyButton");
      button.classList.add("is-loading");
      state.loadingIndex += 1;
      qs("loadingLine").textContent = payload.loading_texts[state.loadingIndex % payload.loading_texts.length] || "";
      window.setTimeout(() => { button.classList.remove("is-loading"); execute(); }, 720);
    }

    function renderSummaryTables() {
      const root = qs("summaryTables");
      root.innerHTML = "";
      payload.summary_tables.forEach((table) => {
        const card = document.createElement("article");
        card.className = "summary-card";
        card.innerHTML = `
          <h3>${table.title}</h3>
          <table class="summary-table">
            <tbody>${table.rows.map((row) => `<tr><td>${row.label}</td><td>${row.value}</td></tr>`).join("")}</tbody>
          </table>
        `;
        root.appendChild(card);
      });
    }

    function renderBars() {
      const root = qs("familyBars");
      root.innerHTML = "";
      payload.family_distribution.forEach((item) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `<div class="bar-label"><span>${item.label}</span><span>${item.count}</span></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(item.ratio * 100, 6)}%"></div></div>`;
        root.appendChild(row);
      });
    }

    function renderTopNodes() {
      const root = qs("topNodes");
      root.innerHTML = "";
      payload.top_nodes.forEach((item) => {
        const row = document.createElement("div");
        row.className = "node-row";
        row.innerHTML = `<strong>${item.name}</strong><span>度数 ${item.degree}</span>`;
        root.appendChild(row);
      });
    }

    function renderPanorama() {
      const style = payload.styles[state.style];
      qs("panoramaIntro").textContent = style.panorama_intro || "查看聚合统计";
      qs("warmLine").textContent = style.warm_line || "";
      qs("sliderLeft").textContent = style.slider_left || "纯模型回答";
      qs("sliderRight").textContent = style.slider_right || "图谱增强回答";
      qs("closingLine").textContent = style.closing || "";
      const isRag = Number(state.slider) >= 100;
      qs("compareMode").textContent = isRag ? (style.slider_right || "图谱增强回答") : (style.slider_left || "纯模型回答");
      qs("compareAccuracy").textContent = formatPercent(isRag ? payload.panorama.rag_accuracy : payload.panorama.baseline_accuracy);
      qs("compareHallucination").textContent = formatPercent(isRag ? payload.panorama.rag_hallucination_rate : payload.panorama.baseline_hallucination_rate);
      renderSummaryTables();
      renderBars();
      renderTopNodes();
      const riskRoot = qs("riskCards");
      riskRoot.innerHTML = "";
      payload.panorama.risk_cards.forEach((card) => {
        const article = document.createElement("article");
        article.className = "risk-card";
        article.innerHTML = `<strong>${card.title}</strong><p>${card.summary}</p><ul>${card.examples.map((example) => `<li>${example}</li>`).join("")}</ul>`;
        riskRoot.appendChild(article);
      });
    }

    function renderCorpusFilters() {
      const categories = ["all", ...Array.from(new Set(payload.corpus.map((item) => item.category)))];
      const labels = ["all", ...Array.from(new Set(payload.corpus.map((item) => item.label)))];
      const catRoot = qs("categoryFilters");
      const labelRoot = qs("labelFilters");
      catRoot.innerHTML = "";
      labelRoot.innerHTML = "";
      categories.forEach((category) => catRoot.appendChild(makeFilterChip(category === "all" ? "全部类别" : category, state.category === category, () => { state.category = category; renderCorpusFilters(); renderCorpusLibrary(); })));
      labels.forEach((label) => labelRoot.appendChild(makeFilterChip(label === "all" ? "全部标签" : label, state.label === label, () => { state.label = label; renderCorpusFilters(); renderCorpusLibrary(); })));
    }

    function getFilteredCorpus() {
      return payload.corpus.filter((item) => (state.category === "all" || item.category === state.category) && (state.label === "all" || item.label === state.label));
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
          <div class="meta">
            <span class="badge soft">${item.category}</span>
            <span class="badge soft">${item.label_full}</span>
            <span class="badge soft">样例 ${item.id}</span>
          </div>
          <h4>${item.text}</h4>
          <div class="actions"><button class="ghost-btn" type="button">填入核查区</button></div>
        `;
        article.querySelector("button").addEventListener("click", () => {
          qs("queryInput").value = item.text;
          window.scrollTo({ top: qs("queryInput").getBoundingClientRect().top + window.scrollY - 120, behavior: "smooth" });
          runVerification(false);
        });
        root.appendChild(article);
      });
      const notes = qs("usageNotes");
      notes.innerHTML = "";
      payload.usage_notes.forEach((text) => {
        const div = document.createElement("div");
        div.className = "footer-note";
        div.textContent = text;
        notes.appendChild(div);
      });
    }

    qs("queryInput").addEventListener("input", renderSuggestions);
    qs("queryInput").addEventListener("focus", renderSuggestions);
    qs("verifyButton").addEventListener("click", () => { closeSuggestions(); runVerification(true); });
    qs("compareSlider").addEventListener("input", (event) => { state.slider = Number(event.target.value); renderPanorama(); });
    document.addEventListener("click", (event) => {
      if (!qs("suggestions").contains(event.target) && event.target !== qs("queryInput")) closeSuggestions();
    });

    renderStyleSwitcher();
    renderActOne();
    renderActTwoText();
    renderPanorama();
    renderCorpusFilters();
    renderCorpusLibrary();
    runVerification(false);
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
