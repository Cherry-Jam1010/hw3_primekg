from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    MENTAL_DISORDER_KEYWORDS,
    RELATION_GROUP_RULES,
    SEED_EXCLUSION_PATTERNS,
    SEED_FAMILY_PATTERNS,
)


STEP1_USECOLS = [
    "relation",
    "display_relation",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
]


def compile_keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    escaped = [re.escape(keyword) for keyword in keywords]
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


def is_keyword_match(series: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, na=False)


def ensure_directories(project_root: Path) -> dict[str, Path]:
    data_processed = project_root / "data" / "processed"
    data_reports = project_root / "data" / "reports"
    results_dir = project_root / "results"
    step2_dir = data_processed / "step2"

    for path in (data_processed, data_reports, results_dir, step2_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "processed": data_processed,
        "reports": data_reports,
        "results": results_dir,
        "step2": step2_dir,
    }


def discover_seed_diseases(primekg_path: Path, chunksize: int) -> pd.DataFrame:
    pattern = compile_keyword_pattern(MENTAL_DISORDER_KEYWORDS)
    disease_rows: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        primekg_path,
        usecols=["x_id", "x_type", "x_name", "y_id", "y_type", "y_name"],
        chunksize=chunksize,
        low_memory=False,
    ):
        x_mask = (chunk["x_type"] == "disease") & is_keyword_match(chunk["x_name"], pattern)
        y_mask = (chunk["y_type"] == "disease") & is_keyword_match(chunk["y_name"], pattern)

        if x_mask.any():
            disease_rows.append(
                chunk.loc[x_mask, ["x_id", "x_name"]]
                .rename(columns={"x_id": "disease_id", "x_name": "disease_name"})
            )
        if y_mask.any():
            disease_rows.append(
                chunk.loc[y_mask, ["y_id", "y_name"]]
                .rename(columns={"y_id": "disease_id", "y_name": "disease_name"})
            )

    if not disease_rows:
        raise ValueError("No mental-health seed diseases were found in PrimeKG.")

    seeds = (
        pd.concat(disease_rows, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values(["disease_name", "disease_id"])
        .reset_index(drop=True)
    )
    return seeds


def annotate_seed_diseases(seeds: pd.DataFrame) -> pd.DataFrame:
    family_rules = {
        family: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for family, patterns in SEED_FAMILY_PATTERNS.items()
    }
    exclusion_rules = [
        re.compile(pattern, flags=re.IGNORECASE) for pattern in SEED_EXCLUSION_PATTERNS
    ]

    annotated = seeds.copy()
    families: list[str] = []
    exclusion_reasons: list[str] = []
    keep_flags: list[bool] = []

    for row in annotated.itertuples(index=False):
        name = str(row.disease_name)
        matched_families = [
            family
            for family, rules in family_rules.items()
            if any(rule.search(name) for rule in rules)
        ]
        matched_exclusions = [
            rule.pattern for rule in exclusion_rules if rule.search(name)
        ]

        families.append("|".join(matched_families))
        exclusion_reasons.append("|".join(matched_exclusions))
        keep_flags.append(bool(matched_families) and not matched_exclusions)

    annotated["seed_family"] = families
    annotated["exclusion_reason"] = exclusion_reasons
    annotated["keep"] = keep_flags
    return annotated.sort_values(
        ["keep", "seed_family", "disease_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def classify_relation_group(x_type: str, y_type: str, display_relation: str) -> str | None:
    for group_name, rule in RELATION_GROUP_RULES.items():
        if (x_type, y_type) in rule["type_pairs"] and display_relation in rule["display_relations"]:
            return group_name
    return None


def extract_subgraph(
    primekg_path: Path,
    seed_disease_ids: set[str],
    chunksize: int,
) -> pd.DataFrame:
    kept_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        primekg_path,
        usecols=STEP1_USECOLS,
        chunksize=chunksize,
        low_memory=False,
    ):
        seed_x = (chunk["x_type"] == "disease") & chunk["x_id"].astype(str).isin(seed_disease_ids)
        seed_y = (chunk["y_type"] == "disease") & chunk["y_id"].astype(str).isin(seed_disease_ids)
        candidate = chunk.loc[seed_x | seed_y].copy()

        if candidate.empty:
            continue

        candidate["relation_group"] = candidate.apply(
            lambda row: classify_relation_group(
                str(row["x_type"]),
                str(row["y_type"]),
                str(row["display_relation"]),
            ),
            axis=1,
        )
        candidate = candidate.loc[candidate["relation_group"].notna()].copy()

        if candidate.empty:
            continue

        disease_disease_mask = candidate["relation_group"] == "disease_disease"
        if disease_disease_mask.any():
            x_in_seed = candidate["x_id"].astype(str).isin(seed_disease_ids)
            y_in_seed = candidate["y_id"].astype(str).isin(seed_disease_ids)
            candidate = candidate.loc[
                (~disease_disease_mask) | (x_in_seed & y_in_seed)
            ].copy()

        if candidate.empty:
            continue

        candidate["seed_side"] = "x"
        candidate.loc[seed_y, "seed_side"] = "y"
        candidate["seed_disease_id"] = candidate["x_id"].astype(str)
        candidate["seed_disease_name"] = candidate["x_name"]
        candidate.loc[candidate["seed_side"] == "y", "seed_disease_id"] = candidate["y_id"].astype(str)
        candidate.loc[candidate["seed_side"] == "y", "seed_disease_name"] = candidate["y_name"]

        kept_chunks.append(candidate)

    if not kept_chunks:
        raise ValueError("No Step 1 subgraph edges were extracted from PrimeKG.")

    subgraph = (
        pd.concat(kept_chunks, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return subgraph


def build_summary(subgraph: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    graph = nx.Graph()

    for row in subgraph.itertuples(index=False):
        graph.add_node(str(row.x_id), name=row.x_name, node_type=row.x_type)
        graph.add_node(str(row.y_id), name=row.y_name, node_type=row.y_type)
        graph.add_edge(
            str(row.x_id),
            str(row.y_id),
            relation_group=row.relation_group,
            display_relation=row.display_relation,
        )

    node_type_counter = Counter()
    for _, data in graph.nodes(data=True):
        node_type_counter[data.get("node_type", "unknown")] += 1

    disease_degree = []
    for node_id, degree in graph.degree():
        node_data = graph.nodes[node_id]
        if node_data.get("node_type") == "disease":
            disease_degree.append(
                {
                    "node_id": node_id,
                    "node_name": node_data.get("name", ""),
                    "degree": int(degree),
                }
            )

    disease_degree.sort(key=lambda item: (-item["degree"], item["node_name"]))

    summary = {
        "seed_disease_count": int(len(seeds)),
        "subgraph_edge_count": int(len(subgraph)),
        "subgraph_node_count": int(graph.number_of_nodes()),
        "connected_component_count": int(nx.number_connected_components(graph)),
        "node_type_distribution": dict(sorted(node_type_counter.items())),
        "relation_group_distribution": subgraph["relation_group"].value_counts().to_dict(),
        "display_relation_distribution": subgraph["display_relation"].value_counts().head(20).to_dict(),
        "seed_family_distribution": (
            seeds["seed_family"].value_counts().to_dict()
            if "seed_family" in seeds.columns
            else {}
        ),
        "top_disease_nodes": disease_degree[:20],
    }
    return summary


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# Step 1 Summary",
        "",
        "## Basic Stats",
        "",
        f"- Seed diseases: {summary['seed_disease_count']}",
        f"- Subgraph edges: {summary['subgraph_edge_count']}",
        f"- Subgraph nodes: {summary['subgraph_node_count']}",
        f"- Connected components: {summary['connected_component_count']}",
        "",
        "## Node Type Distribution",
        "",
    ]

    for node_type, count in summary["node_type_distribution"].items():
        lines.append(f"- {node_type}: {count}")

    lines.extend(["", "## Relation Group Distribution", ""])
    for relation_group, count in summary["relation_group_distribution"].items():
        lines.append(f"- {relation_group}: {count}")

    if summary.get("seed_family_distribution"):
        lines.extend(["", "## Seed Family Distribution", ""])
        for family, count in summary["seed_family_distribution"].items():
            lines.append(f"- {family}: {count}")

    lines.extend(["", "## Top Disease Nodes", ""])
    for item in summary["top_disease_nodes"][:10]:
        lines.append(f"- {item['node_name']} ({item['degree']})")

    return "\n".join(lines) + "\n"


def normalize_relation_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def make_entity_key(node_type: str, node_id: str) -> str:
    return f"{node_type}::{node_id}"


def build_step2_triples(
    subgraph: pd.DataFrame,
    refined_seeds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    refined_seed_map = (
        refined_seeds.assign(disease_id=refined_seeds["disease_id"].astype(str))
        .set_index("disease_id")[["disease_name", "seed_family"]]
        .to_dict("index")
    )

    triples = subgraph.copy()
    triples["head"] = triples.apply(
        lambda row: make_entity_key(str(row["x_type"]), str(row["x_id"])),
        axis=1,
    )
    triples["tail"] = triples.apply(
        lambda row: make_entity_key(str(row["y_type"]), str(row["y_id"])),
        axis=1,
    )
    triples["relation_label"] = triples["display_relation"].astype(str).map(normalize_relation_label)
    triples["head_label"] = triples["x_name"]
    triples["tail_label"] = triples["y_name"]

    entity_rows: list[dict] = []
    seen_entities: set[str] = set()
    for row in triples.itertuples(index=False):
        head_key = row.head
        tail_key = row.tail
        if head_key not in seen_entities:
            head_seed = refined_seed_map.get(str(row.x_id))
            entity_rows.append(
                {
                    "entity": head_key,
                    "node_id": str(row.x_id),
                    "node_name": row.x_name,
                    "node_type": row.x_type,
                    "node_source": row.x_source,
                    "is_seed_disease": bool(head_seed),
                    "seed_family": head_seed["seed_family"] if head_seed else "",
                }
            )
            seen_entities.add(head_key)
        if tail_key not in seen_entities:
            tail_seed = refined_seed_map.get(str(row.y_id))
            entity_rows.append(
                {
                    "entity": tail_key,
                    "node_id": str(row.y_id),
                    "node_name": row.y_name,
                    "node_type": row.y_type,
                    "node_source": row.y_source,
                    "is_seed_disease": bool(tail_seed),
                    "seed_family": tail_seed["seed_family"] if tail_seed else "",
                }
            )
            seen_entities.add(tail_key)

    entities = (
        pd.DataFrame(entity_rows)
        .sort_values(["node_type", "node_name", "node_id"])
        .reset_index(drop=True)
    )
    relations = (
        triples[["relation_label", "display_relation", "relation_group"]]
        .drop_duplicates()
        .sort_values(["relation_group", "relation_label"])
        .reset_index(drop=True)
    )
    export_triples = triples[
        [
            "head",
            "relation_label",
            "tail",
            "head_label",
            "display_relation",
            "tail_label",
            "relation_group",
            "seed_disease_id",
            "seed_disease_name",
        ]
    ].rename(columns={"relation_label": "relation"})
    return export_triples, entities, relations


def split_triples(triples: pd.DataFrame, random_state: int = 42) -> dict[str, pd.DataFrame]:
    split_frames = {"train": [], "valid": [], "test": []}

    for _, group in triples.groupby("relation", sort=True):
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n_rows = len(group)

        if n_rows < 10:
            train_end = max(n_rows - 2, 1)
            valid_end = min(train_end + 1, n_rows - 1) if n_rows > 2 else train_end
            train_group = group.iloc[:train_end]
            valid_group = group.iloc[train_end:valid_end]
            test_group = group.iloc[valid_end:]
        else:
            train_group, temp_group = train_test_split(
                group,
                test_size=0.2,
                random_state=random_state,
                shuffle=True,
            )
            valid_group, test_group = train_test_split(
                temp_group,
                test_size=0.5,
                random_state=random_state,
                shuffle=True,
            )

        split_frames["train"].append(train_group)
        split_frames["valid"].append(valid_group)
        split_frames["test"].append(test_group)

    return {
        split_name: pd.concat(frames, ignore_index=True).sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(drop=True)
        for split_name, frames in split_frames.items()
    }


def enforce_train_coverage(split_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train_df = split_map["train"].copy()
    valid_df = split_map["valid"].copy()
    test_df = split_map["test"].copy()

    changed = True
    while changed:
        changed = False
        train_entities = set(train_df["head"]) | set(train_df["tail"])
        train_relations = set(train_df["relation"])

        for split_name, split_df in (("valid", valid_df), ("test", test_df)):
            coverage_mask = (
                split_df["head"].isin(train_entities)
                & split_df["tail"].isin(train_entities)
                & split_df["relation"].isin(train_relations)
            )
            missing_df = split_df.loc[~coverage_mask].copy()
            if not missing_df.empty:
                train_df = pd.concat([train_df, missing_df], ignore_index=True)
                if split_name == "valid":
                    valid_df = split_df.loc[coverage_mask].reset_index(drop=True)
                else:
                    test_df = split_df.loc[coverage_mask].reset_index(drop=True)
                changed = True

    return {
        "train": train_df.drop_duplicates().reset_index(drop=True),
        "valid": valid_df.drop_duplicates().reset_index(drop=True),
        "test": test_df.drop_duplicates().reset_index(drop=True),
    }


def export_step2_datasets(
    subgraph: pd.DataFrame,
    refined_seeds: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    triples, entities, relations = build_step2_triples(
        subgraph=subgraph,
        refined_seeds=refined_seeds,
    )
    split_map = split_triples(triples)
    split_map = enforce_train_coverage(split_map)

    triples[["head", "relation", "tail"]].to_csv(
        output_dir / "triples.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    triples.to_csv(output_dir / "triples_with_labels.csv", index=False)
    entities.to_csv(output_dir / "entities.csv", index=False)
    relations.to_csv(output_dir / "relations.csv", index=False)
    refined_seeds.to_csv(output_dir / "disease_labels.csv", index=False)

    for split_name, split_df in split_map.items():
        split_df[["head", "relation", "tail"]].to_csv(
            output_dir / f"{split_name}.tsv",
            sep="\t",
            index=False,
            header=False,
        )
        split_df.to_csv(output_dir / f"{split_name}_with_labels.csv", index=False)

    return {
        "triple_count": int(len(triples)),
        "entity_count": int(len(entities)),
        "relation_count": int(len(relations)),
        "train_count": int(len(split_map["train"])),
        "valid_count": int(len(split_map["valid"])),
        "test_count": int(len(split_map["test"])),
    }


def run_step1(primekg_path: Path, project_root: Path, chunksize: int = 200_000) -> None:
    primekg_path = primekg_path.resolve()
    project_root = project_root.resolve()

    if not primekg_path.exists():
        raise FileNotFoundError(f"PrimeKG file not found: {primekg_path}")

    paths = ensure_directories(project_root)

    print(f"[Step 1] PrimeKG input: {primekg_path}")
    print("[Step 1] Discovering seed mental-health diseases...")
    seed_candidates = discover_seed_diseases(primekg_path=primekg_path, chunksize=chunksize)
    annotated_seeds = annotate_seed_diseases(seed_candidates)
    seeds = annotated_seeds.loc[annotated_seeds["keep"]].copy()

    seed_ids = set(seeds["disease_id"].astype(str))
    print(f"[Step 1] Found {len(seed_ids)} seed diseases.")
    print("[Step 1] Extracting disease / phenotype / drug / disease edges...")
    subgraph = extract_subgraph(
        primekg_path=primekg_path,
        seed_disease_ids=seed_ids,
        chunksize=chunksize,
    )

    summary = build_summary(subgraph=subgraph, seeds=seeds)

    subgraph_path = paths["processed"] / "mental_health_subgraph.csv"
    seeds_path = paths["processed"] / "mental_health_seed_diseases.csv"
    seed_candidates_path = paths["processed"] / "mental_health_seed_candidates.csv"
    summary_json_path = paths["reports"] / "step1_summary.json"
    summary_md_path = paths["reports"] / "step1_summary.md"

    subgraph.to_csv(subgraph_path, index=False)
    seeds.to_csv(seeds_path, index=False)
    annotated_seeds.to_csv(seed_candidates_path, index=False)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    print(f"[Step 1] Saved subgraph: {subgraph_path}")
    print(f"[Step 1] Saved seed diseases: {seeds_path}")
    print(f"[Step 1] Saved seed candidates: {seed_candidates_path}")
    print(f"[Step 1] Saved summary: {summary_md_path}")
    print(
        "[Step 1] Done. "
        f"Edges={summary['subgraph_edge_count']}, "
        f"Nodes={summary['subgraph_node_count']}, "
        f"Seeds={summary['seed_disease_count']}"
    )


def run_step2_prep(project_root: Path) -> None:
    project_root = project_root.resolve()
    paths = ensure_directories(project_root)
    subgraph_path = paths["processed"] / "mental_health_subgraph.csv"
    seeds_path = paths["processed"] / "mental_health_seed_diseases.csv"

    if not subgraph_path.exists() or not seeds_path.exists():
        raise FileNotFoundError(
            "Step 2 preparation needs Step 1 outputs first. "
            "Run `python primekg_mental_health_project/main.py step1`."
        )

    print(f"[Step 2] Loading subgraph: {subgraph_path}")
    subgraph = pd.read_csv(subgraph_path, low_memory=False)
    seeds = pd.read_csv(seeds_path, low_memory=False)

    stats = export_step2_datasets(
        subgraph=subgraph,
        refined_seeds=seeds,
        output_dir=paths["step2"],
    )

    step2_summary_path = paths["reports"] / "step2_dataset_summary.json"
    step2_summary = {
        **stats,
        "source_subgraph": str(subgraph_path),
        "source_seed_file": str(seeds_path),
    }
    step2_summary_path.write_text(
        json.dumps(step2_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Step 2] Saved dataset directory: {paths['step2']}")
    print(
        "[Step 2] Done. "
        f"Triples={stats['triple_count']}, "
        f"Entities={stats['entity_count']}, "
        f"Relations={stats['relation_count']}, "
        f"Train/Valid/Test={stats['train_count']}/{stats['valid_count']}/{stats['test_count']}"
    )
