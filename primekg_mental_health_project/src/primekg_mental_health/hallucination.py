from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class ClaimRecord:
    sample_id: str
    label: str
    false_type: str
    relation_group: str
    display_relation: str
    x_id: str
    x_type: str
    x_name: str
    y_id: str
    y_type: str
    y_name: str
    seed_disease_id: str
    seed_disease_name: str
    claim_text: str
    prompt_text: str


def ensure_hallucination_directories(project_root: Path) -> dict[str, Path]:
    processed_root = project_root / "data" / "processed" / "hallucination"
    reports_root = project_root / "data" / "reports"
    results_root = project_root / "results" / "hallucination"
    for path in (processed_root, reports_root, results_root):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "processed": processed_root,
        "reports": reports_root,
        "results": results_root,
    }


def load_subgraph(project_root: Path) -> pd.DataFrame:
    subgraph_path = project_root / "data" / "processed" / "mental_health_subgraph.csv"
    if not subgraph_path.exists():
        raise FileNotFoundError("Step 4 needs Step 1 output mental_health_subgraph.csv.")
    return pd.read_csv(subgraph_path, low_memory=False, dtype=str).fillna("")


def relation_statement(row: pd.Series, *, false_type: str = "real") -> str:
    relation_group = row["relation_group"]
    relation = row["display_relation"]
    x_name = row["x_name"]
    y_name = row["y_name"]

    disease_name = x_name
    other_name = y_name
    if relation_group == "disease_drug":
        if row["x_type"] == "disease":
            disease_name = row["x_name"]
            other_name = row["y_name"]
        else:
            disease_name = row["y_name"]
            other_name = row["x_name"]
    elif relation_group == "disease_phenotype":
        if row["x_type"] == "disease":
            disease_name = row["x_name"]
            other_name = row["y_name"]
        else:
            disease_name = row["y_name"]
            other_name = row["x_name"]

    if relation == "indication":
        if false_type == "polarity_flip":
            return f"很多短视频会说，{other_name} 其实不适合用于 {disease_name}，甚至属于禁忌。"
        return f"很多科普会说，{other_name} 可以用于 {disease_name}。"
    if relation == "contraindication":
        if false_type == "polarity_flip":
            return f"有些博主会说，{other_name} 其实可以放心用于 {disease_name}。"
        return f"很多科普会提醒，{other_name} 对 {disease_name} 属于禁忌或不推荐。"
    if relation == "off-label use":
        if false_type == "polarity_flip":
            return f"有人会直接断言，{other_name} 是 {disease_name} 的标准适应证用药。"
        return f"有些内容会说，{other_name} 有时会被用于 {disease_name} 的非标准适应证场景。"
    if relation == "phenotype present":
        if false_type == "polarity_flip":
            return f"网上有人说，{disease_name} 一般不会表现出 {other_name}。"
        return f"很多科普会说，{disease_name} 常见表现之一是 {other_name}。"
    if relation == "phenotype absent":
        if false_type == "polarity_flip":
            return f"有些说法会把 {other_name} 讲成 {disease_name} 的典型表现。"
        return f"一些资料会强调，{disease_name} 通常并不表现为 {other_name}。"
    if relation == "parent-child":
        if false_type == "hierarchy_error":
            return f"短视频里有人会说，{y_name} 可以被归到 {x_name} 这个大类下面。"
        return f"很多资料会把 {y_name} 视为 {x_name} 的下位类型或更具体的类别。"
    return f"有人声称，{x_name} 和 {y_name} 之间存在 {relation} 关系。"


def build_prompt_text(claim_text: str) -> str:
    return (
        "请判断下面这句精神健康科普说法是否受到 PrimeKG 子图支持，"
        "只回答 supported / unsupported / unknown，并给一句简短理由：\n"
        f"{claim_text}"
    )


def sample_real_claims(subgraph: pd.DataFrame, per_group: int) -> list[ClaimRecord]:
    records: list[ClaimRecord] = []
    for relation_group, group_df in subgraph.groupby("relation_group", sort=True):
        group_df = group_df.sample(
            n=min(per_group, len(group_df)),
            random_state=RANDOM_SEED,
        )
        for i, (_, row) in enumerate(group_df.iterrows(), start=1):
            claim_text = relation_statement(row, false_type="real")
            records.append(
                ClaimRecord(
                    sample_id=f"real_{relation_group}_{i:02d}",
                    label="real",
                    false_type="real",
                    relation_group=relation_group,
                    display_relation=row["display_relation"],
                    x_id=row["x_id"],
                    x_type=row["x_type"],
                    x_name=row["x_name"],
                    y_id=row["y_id"],
                    y_type=row["y_type"],
                    y_name=row["y_name"],
                    seed_disease_id=row["seed_disease_id"],
                    seed_disease_name=row["seed_disease_name"],
                    claim_text=claim_text,
                    prompt_text=build_prompt_text(claim_text),
                )
            )
    return records


def build_existing_triplet_keys(subgraph: pd.DataFrame) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            row["x_id"],
            row["x_type"],
            row["display_relation"],
            row["y_id"],
            row["y_type"],
        )
        for _, row in subgraph.iterrows()
    }


def fabricate_tail(
    source_row: pd.Series,
    candidate_pool: pd.DataFrame,
    existing_keys: set[tuple[str, str, str, str, str]],
) -> pd.Series | None:
    x_id = source_row["x_id"]
    x_type = source_row["x_type"]
    relation = source_row["display_relation"]
    y_type = source_row["y_type"]

    candidates = candidate_pool.loc[
        (candidate_pool["y_type"] == y_type) & (candidate_pool["y_id"] != source_row["y_id"])
    ].copy()
    candidates = candidates.sample(frac=1.0, random_state=RANDOM_SEED)

    for _, candidate in candidates.iterrows():
        key = (x_id, x_type, relation, candidate["y_id"], y_type)
        if key not in existing_keys:
            fabricated = source_row.copy()
            fabricated["y_id"] = candidate["y_id"]
            fabricated["y_name"] = candidate["y_name"]
            return fabricated
    return None


def sample_false_claims(subgraph: pd.DataFrame, per_type: int) -> list[ClaimRecord]:
    existing_keys = build_existing_triplet_keys(subgraph)
    records: list[ClaimRecord] = []

    polarity_candidates = subgraph.loc[
        subgraph["display_relation"].isin(["indication", "contraindication", "phenotype present", "phenotype absent"])
    ].sample(frac=1.0, random_state=RANDOM_SEED)
    hierarchy_candidates = subgraph.loc[
        subgraph["display_relation"] == "parent-child"
    ].sample(frac=1.0, random_state=RANDOM_SEED)
    fabricated_candidates = subgraph.sample(frac=1.0, random_state=RANDOM_SEED)

    polarity_map = {
        "indication": "contraindication",
        "contraindication": "indication",
        "phenotype present": "phenotype absent",
        "phenotype absent": "phenotype present",
    }

    polarity_count = 0
    for _, row in polarity_candidates.iterrows():
        if polarity_count >= per_type:
            break
        false_row = row.copy()
        false_row["display_relation"] = polarity_map[row["display_relation"]]
        key = (
            false_row["x_id"],
            false_row["x_type"],
            false_row["display_relation"],
            false_row["y_id"],
            false_row["y_type"],
        )
        if key in existing_keys:
            continue
        claim_text = relation_statement(false_row, false_type="polarity_flip")
        polarity_count += 1
        records.append(
            ClaimRecord(
                sample_id=f"false_polarity_{polarity_count:02d}",
                label="false",
                false_type="polarity_flip",
                relation_group=false_row["relation_group"],
                display_relation=false_row["display_relation"],
                x_id=false_row["x_id"],
                x_type=false_row["x_type"],
                x_name=false_row["x_name"],
                y_id=false_row["y_id"],
                y_type=false_row["y_type"],
                y_name=false_row["y_name"],
                seed_disease_id=false_row["seed_disease_id"],
                seed_disease_name=false_row["seed_disease_name"],
                claim_text=claim_text,
                prompt_text=build_prompt_text(claim_text),
            )
        )

    hierarchy_count = 0
    disease_pool = subgraph.loc[subgraph["x_type"] == "disease", ["x_id", "x_name"]].drop_duplicates()
    disease_pool = pd.concat(
        [
            disease_pool,
            subgraph.loc[subgraph["y_type"] == "disease", ["y_id", "y_name"]]
            .rename(columns={"y_id": "x_id", "y_name": "x_name"})
            .drop_duplicates(),
        ],
        ignore_index=True,
    ).drop_duplicates()
    for _, row in hierarchy_candidates.iterrows():
        if hierarchy_count >= per_type:
            break

        false_row = None

        reversed_row = row.copy()
        reversed_row["x_id"], reversed_row["y_id"] = row["y_id"], row["x_id"]
        reversed_row["x_type"], reversed_row["y_type"] = row["y_type"], row["x_type"]
        reversed_row["x_name"], reversed_row["y_name"] = row["y_name"], row["x_name"]
        reversed_key = (
            reversed_row["x_id"],
            reversed_row["x_type"],
            reversed_row["display_relation"],
            reversed_row["y_id"],
            reversed_row["y_type"],
        )
        if reversed_key not in existing_keys:
            false_row = reversed_row
        else:
            for _, candidate in disease_pool.sample(frac=1.0, random_state=RANDOM_SEED).iterrows():
                if candidate["x_id"] in {row["x_id"], row["y_id"]}:
                    continue
                candidate_key = (
                    row["x_id"],
                    row["x_type"],
                    row["display_relation"],
                    str(candidate["x_id"]),
                    "disease",
                )
                if candidate_key not in existing_keys:
                    false_row = row.copy()
                    false_row["y_id"] = str(candidate["x_id"])
                    false_row["y_type"] = "disease"
                    false_row["y_name"] = candidate["x_name"]
                    break

        if false_row is None:
            continue

        claim_text = relation_statement(false_row, false_type="hierarchy_error")
        hierarchy_count += 1
        records.append(
            ClaimRecord(
                sample_id=f"false_hierarchy_{hierarchy_count:02d}",
                label="false",
                false_type="hierarchy_error",
                relation_group=false_row["relation_group"],
                display_relation=false_row["display_relation"],
                x_id=false_row["x_id"],
                x_type=false_row["x_type"],
                x_name=false_row["x_name"],
                y_id=false_row["y_id"],
                y_type=false_row["y_type"],
                y_name=false_row["y_name"],
                seed_disease_id=false_row["seed_disease_id"],
                seed_disease_name=false_row["seed_disease_name"],
                claim_text=claim_text,
                prompt_text=build_prompt_text(claim_text),
            )
        )

    fabricated_count = 0
    for _, row in fabricated_candidates.iterrows():
        if fabricated_count >= per_type:
            break
        false_row = fabricate_tail(
            source_row=row,
            candidate_pool=subgraph,
            existing_keys=existing_keys,
        )
        if false_row is None:
            continue
        claim_text = relation_statement(false_row, false_type="fabricated")
        fabricated_count += 1
        records.append(
            ClaimRecord(
                sample_id=f"false_fabricated_{fabricated_count:02d}",
                label="false",
                false_type="fabricated",
                relation_group=false_row["relation_group"],
                display_relation=false_row["display_relation"],
                x_id=false_row["x_id"],
                x_type=false_row["x_type"],
                x_name=false_row["x_name"],
                y_id=false_row["y_id"],
                y_type=false_row["y_type"],
                y_name=false_row["y_name"],
                seed_disease_id=false_row["seed_disease_id"],
                seed_disease_name=false_row["seed_disease_name"],
                claim_text=claim_text,
                prompt_text=build_prompt_text(claim_text),
            )
        )

    return records


def claims_to_dataframe(records: list[ClaimRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def simulate_llm_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    profiles = {
        "real": 0.78,
        "polarity_flip": 0.58,
        "hierarchy_error": 0.62,
        "fabricated": 0.86,
    }
    rows = []
    rng = random.Random(RANDOM_SEED)
    for _, row in dataset.iterrows():
        false_type = row["false_type"]
        target_label = "supported" if row["label"] == "real" else "unsupported"
        correct_prob = profiles.get(false_type, 0.7)
        is_correct = rng.random() < correct_prob
        if is_correct:
            prediction = target_label
        else:
            prediction = "unsupported" if target_label == "supported" else "supported"
        rows.append(
            {
                **row.to_dict(),
                "mock_model": "simulated_general_llm",
                "predicted_label": prediction,
                "target_label": target_label,
                "is_correct": is_correct,
                "note": "Simulated baseline, not a real LLM API call.",
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> dict:
    overall_accuracy = float(predictions["is_correct"].mean()) if len(predictions) else 0.0
    summary = {
        "sample_count": int(len(predictions)),
        "accuracy": overall_accuracy,
        "hallucination_rate": 1.0 - overall_accuracy,
        "by_label": {},
        "by_false_type": {},
        "note": "This summary is based on the simulated baseline, not a live LLM.",
    }

    for label, group in predictions.groupby("label", sort=True):
        summary["by_label"][label] = {
            "count": int(len(group)),
            "accuracy": float(group["is_correct"].mean()),
        }

    for false_type, group in predictions.groupby("false_type", sort=True):
        summary["by_false_type"][false_type] = {
            "count": int(len(group)),
            "accuracy": float(group["is_correct"].mean()),
        }

    return summary


def plot_hallucination_summary(summary: dict, output_path: Path) -> None:
    by_false_type = summary.get("by_false_type", {})
    labels = list(by_false_type.keys())
    values = [by_false_type[label]["accuracy"] * 100 for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Simulated Hallucination Detection Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_hallucination_pipeline(
    project_root: Path,
    *,
    real_per_group: int = 10,
    false_per_type: int = 10,
) -> Path:
    project_root = project_root.resolve()
    paths = ensure_hallucination_directories(project_root)
    subgraph = load_subgraph(project_root)

    real_records = sample_real_claims(subgraph=subgraph, per_group=real_per_group)
    false_records = sample_false_claims(subgraph=subgraph, per_type=false_per_type)
    dataset = claims_to_dataframe(real_records + false_records)
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    dataset_csv = paths["processed"] / "step4_hallucination_dataset.csv"
    dataset_json = paths["processed"] / "step4_hallucination_dataset.json"
    dataset.to_csv(dataset_csv, index=False)
    dataset.to_json(dataset_json, orient="records", force_ascii=False, indent=2)

    predictions = simulate_llm_predictions(dataset)
    predictions_csv = paths["reports"] / "step4_simulated_llm_predictions.csv"
    predictions.to_csv(predictions_csv, index=False)

    summary = summarize_predictions(predictions)
    summary_json = paths["reports"] / "step4_simulated_llm_summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_summary = {
        "total_samples": int(len(dataset)),
        "real_samples": int((dataset["label"] == "real").sum()),
        "false_samples": int((dataset["label"] == "false").sum()),
        "by_relation_group": dataset["relation_group"].value_counts().to_dict(),
        "by_false_type": dataset["false_type"].value_counts().to_dict(),
    }
    dataset_summary_json = paths["reports"] / "step4_dataset_summary.json"
    dataset_summary_json.write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_hallucination_summary(summary, paths["results"] / "step4_simulated_llm_accuracy.png")
    return dataset_csv
