from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def ensure_classification_directories(project_root: Path) -> dict[str, Path]:
    results_root = project_root / "results" / "classification"
    reports_root = project_root / "data" / "reports"
    processed_root = project_root / "data" / "processed" / "classification"
    for path in (results_root, reports_root, processed_root):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "results": results_root,
        "reports": reports_root,
        "processed": processed_root,
    }


def load_entity_mapping(model_dir: Path) -> pd.DataFrame:
    mapping_path = model_dir / "training_triples" / "entity_to_id.tsv.gz"
    mapping = pd.read_csv(mapping_path, sep="\t")
    return mapping.rename(columns={"label": "entity_label", "id": "entity_index"})


def load_entity_embeddings(model_dir: Path) -> torch.Tensor:
    model = torch.load(
        model_dir / "trained_model.pkl",
        map_location="cpu",
        weights_only=False,
    )
    embeddings = model.entity_representations[0](indices=None).detach().cpu()
    return embeddings


def embedding_to_feature_vector(value) -> list[float]:
    vector = np.asarray(value)
    if np.iscomplexobj(vector):
        vector = np.concatenate([vector.real, vector.imag], axis=0)
    return vector.astype(float).tolist()


def load_labeled_diseases(project_root: Path) -> pd.DataFrame:
    labels_path = project_root / "data" / "processed" / "mental_health_seed_diseases.csv"
    labels = pd.read_csv(labels_path, dtype={"disease_id": str})
    labels["entity_label"] = labels["disease_id"].map(lambda value: f"disease::{value}")
    return labels


def build_embedding_dataset(
    project_root: Path,
    model_name: str,
    min_class_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_dir = project_root / "results" / "kge" / model_name
    labels = load_labeled_diseases(project_root)
    mapping = load_entity_mapping(model_dir)
    embeddings = load_entity_embeddings(model_dir)

    dataset = labels.merge(mapping, on="entity_label", how="left")
    missing_df = dataset.loc[dataset["entity_index"].isna()].copy()
    present_df = dataset.loc[dataset["entity_index"].notna()].copy()

    present_df["entity_index"] = present_df["entity_index"].astype(int)
    present_df["embedding"] = present_df["entity_index"].map(
        lambda idx: embedding_to_feature_vector(embeddings[idx].numpy())
    )

    family_counts = present_df["seed_family"].value_counts()
    keep_families = family_counts[family_counts >= min_class_size].index.tolist()
    filtered = present_df.loc[present_df["seed_family"].isin(keep_families)].copy()
    excluded = present_df.loc[~present_df["seed_family"].isin(keep_families)].copy()
    if not missing_df.empty:
        missing_df = missing_df.assign(exclusion_reason="missing_trained_entity")
    if not excluded.empty:
        excluded = excluded.assign(exclusion_reason="class_too_small")
    return (
        filtered.reset_index(drop=True),
        excluded.reset_index(drop=True),
        missing_df.reset_index(drop=True),
    )


def run_single_classification(
    dataset: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x = pd.DataFrame(dataset["embedding"].tolist()).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset["seed_family"])
    class_names = label_encoder.classes_.tolist()

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    predicted = cross_val_predict(classifier, x, y, cv=splitter)

    metrics = {
        "embedding_model": model_name,
        "sample_count": int(len(dataset)),
        "class_count": int(len(class_names)),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "weighted_f1": float(f1_score(y, predicted, average="weighted")),
        "cv_splits": int(n_splits),
        "classes": class_names,
    }

    report_df = (
        pd.DataFrame(classification_report(y, predicted, target_names=class_names, output_dict=True))
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )
    confusion_df = pd.DataFrame(
        confusion_matrix(y, predicted),
        index=class_names,
        columns=class_names,
    )
    return metrics, report_df, confusion_df


def save_confusion_matrix(confusion_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_classification_experiment(
    project_root: Path,
    *,
    min_class_size: int = 3,
) -> Path:
    project_root = project_root.resolve()
    paths = ensure_classification_directories(project_root)

    summary_rows = []
    best_confusion = None
    best_model_name = None
    best_macro_f1 = float("-inf")

    for model_name in ("rotate", "transe"):
        filtered, excluded, missing = build_embedding_dataset(
            project_root=project_root,
            model_name=model_name,
            min_class_size=min_class_size,
        )
        min_count = int(filtered["seed_family"].value_counts().min())
        n_splits = min(5, min_count)
        metrics, report_df, confusion_df = run_single_classification(
            filtered,
            model_name=model_name,
            n_splits=n_splits,
        )
        metrics["excluded_class_count"] = int(excluded["seed_family"].nunique())
        metrics["excluded_sample_count"] = int(len(excluded))
        metrics["missing_entity_sample_count"] = int(len(missing))
        summary_rows.append(metrics)

        filtered.to_json(
            paths["processed"] / f"{model_name}_classification_dataset.json",
            orient="records",
            force_ascii=False,
            indent=2,
        )
        report_df.to_csv(paths["reports"] / f"step3_{model_name}_classification_report.csv", index=False)
        confusion_df.to_csv(paths["reports"] / f"step3_{model_name}_confusion_matrix.csv")
        excluded.to_csv(paths["reports"] / f"step3_{model_name}_excluded_labels.csv", index=False)
        missing.to_csv(paths["reports"] / f"step3_{model_name}_missing_entity_labels.csv", index=False)
        save_confusion_matrix(
            confusion_df=confusion_df,
            output_path=paths["results"] / f"{model_name}_confusion_matrix.png",
            title=f"{model_name.upper()} disease-family classification",
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_confusion = confusion_df
            best_model_name = model_name

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_csv = paths["reports"] / "step3_classification_summary.csv"
    summary_json = paths["reports"] / "step3_classification_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    if best_confusion is not None and best_model_name is not None:
        save_confusion_matrix(
            confusion_df=best_confusion,
            output_path=paths["results"] / "best_model_confusion_matrix.png",
            title=f"Best model: {best_model_name.upper()}",
        )

    return summary_csv
