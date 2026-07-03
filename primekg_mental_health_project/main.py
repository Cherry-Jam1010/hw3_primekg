import argparse
from pathlib import Path

from src.primekg_mental_health.demo_showcase import run_demo_pipeline
from src.primekg_mental_health.hallucination import run_hallucination_pipeline
from src.primekg_mental_health.pipeline import run_step1, run_step2_prep
from src.primekg_mental_health.classification import run_classification_experiment
from src.primekg_mental_health.rag import run_rag_experiment
from src.primekg_mental_health.training import train_kge_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PrimeKG mental health project pipeline"
    )
    parser.add_argument(
        "step",
        choices=["step1", "step2_prep", "step2_train", "step3_classify", "step4_hallucination", "step5_rag", "step6_demo"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--primekg",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "primekg_raw.csv",
        help="Path to the PrimeKG CSV file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory for outputs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunk size for streaming PrimeKG.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for KGE models.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension for KGE models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for KGE training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for KGE training.",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=3,
        help="Minimum number of samples per class for Step 3 classification.",
    )
    parser.add_argument(
        "--real-per-group",
        type=int,
        default=10,
        help="Number of real claims to sample per relation group in Step 4.",
    )
    parser.add_argument(
        "--false-per-type",
        type=int,
        default=10,
        help="Number of false claims to generate per false type in Step 4.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval size for Step 5 RAG experiment.",
    )
    parser.add_argument(
        "--demo-sample-limit",
        type=int,
        default=12,
        help="Number of benchmark claims to surface in the Step 6 demo.",
    )
    parser.add_argument(
        "--serve-demo",
        action="store_true",
        help="Serve the generated Step 6 demo locally after building it.",
    )
    parser.add_argument(
        "--demo-host",
        type=str,
        default="127.0.0.1",
        help="Host address for the optional Step 6 local server.",
    )
    parser.add_argument(
        "--demo-port",
        type=int,
        default=8765,
        help="Port for the optional Step 6 local server.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.step == "step1":
        run_step1(
            primekg_path=args.primekg,
            project_root=args.project_root,
            chunksize=args.chunksize,
        )
    elif args.step == "step2_prep":
        run_step2_prep(project_root=args.project_root)
    elif args.step == "step2_train":
        comparison_csv = train_kge_models(
            project_root=args.project_root,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(f"KGE comparison saved to: {comparison_csv}")
    elif args.step == "step3_classify":
        summary_csv = run_classification_experiment(
            project_root=args.project_root,
            min_class_size=args.min_class_size,
        )
        print(f"Classification summary saved to: {summary_csv}")
    elif args.step == "step4_hallucination":
        dataset_csv = run_hallucination_pipeline(
            project_root=args.project_root,
            real_per_group=args.real_per_group,
            false_per_type=args.false_per_type,
        )
        print(f"Hallucination dataset saved to: {dataset_csv}")
    elif args.step == "step5_rag":
        summary_csv = run_rag_experiment(
            project_root=args.project_root,
            top_k=args.top_k,
        )
        print(f"RAG comparison summary saved to: {summary_csv}")
    elif args.step == "step6_demo":
        summary_json = run_demo_pipeline(
            project_root=args.project_root,
            sample_limit=args.demo_sample_limit,
            serve=args.serve_demo,
            host=args.demo_host,
            port=args.demo_port,
        )
        print(f"Demo summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
