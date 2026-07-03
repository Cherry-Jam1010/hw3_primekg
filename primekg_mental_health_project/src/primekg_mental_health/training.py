from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def ensure_training_directories(project_root: Path) -> dict[str, Path]:
    results_root = project_root / "results" / "kge"
    reports_root = project_root / "data" / "reports"
    for path in (results_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "results": results_root,
        "reports": reports_root,
    }


def load_triples(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
    )


def load_factories(step2_dir: Path) -> tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    train_df = load_triples(step2_dir / "train.tsv")
    valid_df = load_triples(step2_dir / "valid.tsv")
    test_df = load_triples(step2_dir / "test.tsv")

    train_tf = TriplesFactory.from_labeled_triples(
        train_df[["head", "relation", "tail"]].to_numpy(dtype=str),
        create_inverse_triples=False,
    )
    valid_tf = TriplesFactory.from_labeled_triples(
        valid_df[["head", "relation", "tail"]].to_numpy(dtype=str),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=False,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        test_df[["head", "relation", "tail"]].to_numpy(dtype=str),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=False,
    )
    return train_tf, valid_tf, test_tf


def collect_metrics(result, model_name: str, config: dict) -> dict:
    metrics = result.metric_results.to_flat_dict()
    return {
        "model": model_name,
        "epochs": config["epochs"],
        "embedding_dim": config["embedding_dim"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "device": config["device"],
        "mr": float(metrics.get("both.realistic.arithmetic_mean_rank", 0.0)),
        "mrr": float(metrics.get("both.realistic.inverse_harmonic_mean_rank", 0.0)),
        "hits_at_1": float(metrics.get("both.realistic.hits_at_1", 0.0)),
        "hits_at_3": float(metrics.get("both.realistic.hits_at_3", 0.0)),
        "hits_at_10": float(metrics.get("both.realistic.hits_at_10", 0.0)),
    }


def train_single_model(
    model_name: str,
    train_tf: TriplesFactory,
    valid_tf: TriplesFactory,
    test_tf: TriplesFactory,
    output_dir: Path,
    *,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "epochs": epochs,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "random_seed": random_seed,
    }

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        optimizer="Adam",
        optimizer_kwargs={"lr": learning_rate},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
        },
        evaluator="RankBasedEvaluator",
        evaluation_kwargs={"batch_size": batch_size},
        device=device,
        random_seed=random_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(output_dir)

    metrics = collect_metrics(result=result, model_name=model_name, config=config)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def train_kge_models(
    project_root: Path,
    *,
    epochs: int = 30,
    embedding_dim: int = 64,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> Path:
    project_root = project_root.resolve()
    step2_dir = project_root / "data" / "processed" / "step2"
    if not step2_dir.exists():
        raise FileNotFoundError(
            "Step 2 prepared data is missing. Run step1 and step2_prep first."
        )

    paths = ensure_training_directories(project_root)
    train_tf, valid_tf, test_tf = load_factories(step2_dir=step2_dir)

    metrics_rows = []
    for model_name in ("TransE", "RotatE"):
        model_output_dir = paths["results"] / model_name.lower()
        metrics = train_single_model(
            model_name=model_name,
            train_tf=train_tf,
            valid_tf=valid_tf,
            test_tf=test_tf,
            output_dir=model_output_dir,
            epochs=epochs,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mrr", ascending=False)
    comparison_csv = paths["reports"] / "step2_model_comparison.csv"
    comparison_json = paths["reports"] / "step2_model_comparison.json"
    metrics_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        metrics_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return comparison_csv
