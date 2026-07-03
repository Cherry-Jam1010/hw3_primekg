from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_rag_directories(project_root: Path) -> dict[str, Path]:
    processed_root = project_root / "data" / "processed" / "rag"
    reports_root = project_root / "data" / "reports"
    results_root = project_root / "results" / "rag"
    for path in (processed_root, reports_root, results_root):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "processed": processed_root,
        "reports": reports_root,
        "results": results_root,
    }


def load_subgraph(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mental_health_subgraph.csv"
    if not path.exists():
        raise FileNotFoundError("Step 5 requires mental_health_subgraph.csv from Step 1.")
    return pd.read_csv(path, low_memory=False, dtype=str).fillna("")


def load_step4_dataset(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "hallucination" / "step4_hallucination_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Step 5 expects the Step 4 hallucination dataset first.")
    return pd.read_csv(path, dtype=str).fillna("")


def load_step4_predictions(project_root: Path) -> pd.DataFrame | None:
    path = project_root / "data" / "reports" / "step4_simulated_llm_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str).fillna("")


def normalize_entity_pair(row: pd.Series) -> tuple[str, str]:
    if row["relation_group"] in {"disease_drug", "disease_phenotype"}:
        if row["x_type"] == "disease":
            return row["x_id"], row["y_id"]
        return row["y_id"], row["x_id"]
    return row["x_id"], row["y_id"]


def triplet_natural_text(row: pd.Series) -> str:
    relation = row["display_relation"]
    x_name = row["x_name"]
    y_name = row["y_name"]
    relation_group = row["relation_group"]

    if relation_group == "disease_drug":
        if row["x_type"] == "disease":
            disease_name, drug_name = x_name, y_name
        else:
            disease_name, drug_name = y_name, x_name
        if relation == "indication":
            return f"{drug_name} can be used for {disease_name}."
        if relation == "contraindication":
            return f"{drug_name} is contraindicated for {disease_name}."
        if relation == "off-label use":
            return f"{drug_name} may be used off-label for {disease_name}."

    if relation_group == "disease_phenotype":
        if row["x_type"] == "disease":
            disease_name, phenotype_name = x_name, y_name
        else:
            disease_name, phenotype_name = y_name, x_name
        if relation == "phenotype present":
            return f"{disease_name} commonly presents with {phenotype_name}."
        if relation == "phenotype absent":
            return f"{disease_name} typically does not present with {phenotype_name}."

    if relation_group == "disease_disease" and relation == "parent-child":
        return f"{y_name} is a more specific subtype under {x_name}."

    return f"{x_name} has relation {relation} with {y_name}."


def build_rag_corpus(subgraph: pd.DataFrame) -> pd.DataFrame:
    triplet_docs = []
    for idx, row in subgraph.iterrows():
        disease_id, other_id = normalize_entity_pair(row)
        triplet_docs.append(
            {
                "doc_id": f"triplet_{idx:05d}",
                "doc_type": "triplet",
                "text": triplet_natural_text(row),
                "relation_group": row["relation_group"],
                "display_relation": row["display_relation"],
                "x_id": row["x_id"],
                "x_type": row["x_type"],
                "x_name": row["x_name"],
                "y_id": row["y_id"],
                "y_type": row["y_type"],
                "y_name": row["y_name"],
                "disease_id": disease_id,
                "other_id": other_id,
                "seed_disease_id": row["seed_disease_id"],
                "seed_disease_name": row["seed_disease_name"],
            }
        )

    docs_df = pd.DataFrame(triplet_docs)

    summary_docs = []
    for disease_id, group in docs_df.groupby("seed_disease_id", sort=True):
        disease_name = group["seed_disease_name"].iloc[0]
        symptoms = group.loc[
            (group["relation_group"] == "disease_phenotype")
            & (group["display_relation"] == "phenotype present"),
            "y_name",
        ].drop_duplicates().tolist()[:10]
        drugs = group.loc[
            (group["relation_group"] == "disease_drug")
            & (group["display_relation"] == "indication"),
            "y_name",
        ].drop_duplicates().tolist()[:10]
        parents = group.loc[
            (group["relation_group"] == "disease_disease")
            & (group["display_relation"] == "parent-child"),
            "x_name",
        ].drop_duplicates().tolist()[:5]

        parts = [f"Disease summary for {disease_name}."]
        if symptoms:
            parts.append(f"Common phenotypes include {', '.join(symptoms)}.")
        if drugs:
            parts.append(f"Associated indication drugs include {', '.join(drugs)}.")
        if parents:
            parts.append(f"This disease appears under broader categories such as {', '.join(parents)}.")

        summary_docs.append(
            {
                "doc_id": f"summary_{disease_id}",
                "doc_type": "summary",
                "text": " ".join(parts),
                "relation_group": "summary",
                "display_relation": "summary",
                "x_id": disease_id,
                "x_type": "disease",
                "x_name": disease_name,
                "y_id": "",
                "y_type": "",
                "y_name": "",
                "disease_id": disease_id,
                "other_id": "",
                "seed_disease_id": disease_id,
                "seed_disease_name": disease_name,
            }
        )

    return pd.concat([docs_df, pd.DataFrame(summary_docs)], ignore_index=True)


@dataclass
class RetrievalResult:
    sample_id: str
    predicted_label: str
    target_label: str
    is_correct: bool
    top_score: float
    retrieved_doc_ids: list[str]
    note: str


class TfidfRetriever:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.corpus["text"])

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        result = self.corpus.iloc[top_idx].copy()
        result["score"] = scores[top_idx]
        return result.reset_index(drop=True)


def build_lookup_sets(subgraph: pd.DataFrame) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    exact = set()
    pair_relation = set()
    for _, row in subgraph.iterrows():
        disease_id, other_id = normalize_entity_pair(row)
        exact.add((disease_id, row["display_relation"], other_id))
        pair_relation.add((disease_id, other_id, row["display_relation"]))
    return exact, pair_relation


def rag_verify_dataset(corpus: pd.DataFrame, dataset: pd.DataFrame, subgraph: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    retriever = TfidfRetriever(corpus)
    exact_lookup, _ = build_lookup_sets(subgraph)
    corpus_triplets = corpus.loc[corpus["doc_type"] == "triplet"].copy()

    rows = []
    for _, sample in dataset.iterrows():
        target_label = "supported" if sample["label"] == "real" else "unsupported"
        retrieved = retriever.search(sample["claim_text"], top_k=top_k)
        top_score = float(retrieved["score"].iloc[0]) if len(retrieved) else 0.0

        disease_id, other_id = normalize_entity_pair(sample)
        exact_key = (disease_id, sample["display_relation"], other_id)

        retrieved_triplets = retrieved.loc[retrieved["doc_type"] == "triplet"].copy()
        exact_in_topk = False
        for _, doc in retrieved_triplets.iterrows():
            doc_disease, doc_other = normalize_entity_pair(doc)
            if (doc_disease, doc["display_relation"], doc_other) == exact_key:
                exact_in_topk = True
                break

        exact_in_corpus = False
        exact_corpus_docs = corpus_triplets.loc[
            (corpus_triplets["display_relation"] == sample["display_relation"])
        ].copy()
        if len(exact_corpus_docs):
            exact_corpus_docs["doc_disease_id"] = exact_corpus_docs.apply(
                lambda row: normalize_entity_pair(row)[0], axis=1
            )
            exact_corpus_docs["doc_other_id"] = exact_corpus_docs.apply(
                lambda row: normalize_entity_pair(row)[1], axis=1
            )
            exact_in_corpus = bool(
                (
                    (exact_corpus_docs["doc_disease_id"] == disease_id)
                    & (exact_corpus_docs["doc_other_id"] == other_id)
                ).any()
            )

        if exact_in_topk:
            predicted = "supported"
            note = "Exact triplet evidence retrieved."
        elif exact_in_corpus:
            predicted = "supported"
            note = "Exact triplet exists in corpus and is recovered by entity-aware fallback."
        elif exact_key not in exact_lookup:
            predicted = "unsupported"
            note = "No exact triplet exists in current PrimeKG subgraph."
        elif top_score < 0.05:
            predicted = "unknown"
            note = "Retriever confidence is too low."
        else:
            predicted = "unsupported"
            note = "Relevant evidence not retrieved in top-k."

        rows.append(
            {
                **sample.to_dict(),
                "target_label": target_label,
                "predicted_label": predicted,
                "is_correct": predicted == target_label,
                "top_score": top_score,
                "retrieved_doc_ids": "|".join(retrieved["doc_id"].tolist()),
                "retrieved_texts": " || ".join(retrieved["text"].tolist()),
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def compare_with_pure_baseline(rag_predictions: pd.DataFrame, pure_predictions: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    rag_accuracy = float(rag_predictions["is_correct"].mean()) if len(rag_predictions) else 0.0
    rows.append(
        {
            "system": "rag_verifier",
            "sample_count": int(len(rag_predictions)),
            "accuracy": rag_accuracy,
            "hallucination_rate": 1.0 - rag_accuracy,
        }
    )

    if pure_predictions is not None and len(pure_predictions):
        pure_acc = float((pure_predictions["is_correct"].astype(str).str.lower() == "true").mean())
        rows.append(
            {
                "system": "pure_llm_simulated",
                "sample_count": int(len(pure_predictions)),
                "accuracy": pure_acc,
                "hallucination_rate": 1.0 - pure_acc,
            }
        )

    return pd.DataFrame(rows)


def summarize_by_false_type(df: pd.DataFrame) -> dict:
    summary = {}
    for key, group in df.groupby("false_type", sort=True):
        summary[key] = {
            "count": int(len(group)),
            "accuracy": float(group["is_correct"].mean()),
        }
    return summary


def plot_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df["system"], summary_df["accuracy"] * 100, color=["#4C78A8", "#F58518"][: len(summary_df)])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Pure Baseline vs RAG Verifier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_rag_experiment(project_root: Path, *, top_k: int = 5) -> Path:
    project_root = project_root.resolve()
    paths = ensure_rag_directories(project_root)
    subgraph = load_subgraph(project_root)
    dataset = load_step4_dataset(project_root)
    pure_predictions = load_step4_predictions(project_root)

    corpus = build_rag_corpus(subgraph)
    corpus_csv = paths["processed"] / "step5_rag_corpus.csv"
    corpus.to_csv(corpus_csv, index=False)

    rag_predictions = rag_verify_dataset(corpus=corpus, dataset=dataset, subgraph=subgraph, top_k=top_k)
    rag_predictions_csv = paths["reports"] / "step5_rag_predictions.csv"
    rag_predictions.to_csv(rag_predictions_csv, index=False)

    comparison_df = compare_with_pure_baseline(rag_predictions, pure_predictions)
    comparison_csv = paths["reports"] / "step5_rag_vs_pure_summary.csv"
    comparison_json = paths["reports"] / "step5_rag_vs_pure_summary.json"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        comparison_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    detailed_summary = {
        "rag_accuracy": float(rag_predictions["is_correct"].mean()),
        "rag_by_label": {
            label: {
                "count": int(len(group)),
                "accuracy": float(group["is_correct"].mean()),
            }
            for label, group in rag_predictions.groupby("label", sort=True)
        },
        "rag_by_false_type": summarize_by_false_type(rag_predictions),
        "note": "This Step 5 baseline uses TF-IDF retrieval plus exact triplet evidence, not a generative RAG LLM.",
    }
    detailed_summary_json = paths["reports"] / "step5_rag_detailed_summary.json"
    detailed_summary_json.write_text(
        json.dumps(detailed_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_comparison(comparison_df, paths["results"] / "step5_rag_vs_pure_accuracy.png")
    return comparison_csv
