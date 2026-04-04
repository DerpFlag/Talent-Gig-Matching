import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluate import evaluate_retrieval_and_rerank


def main() -> None:
    result = evaluate_retrieval_and_rerank(output_path="data/artifacts/eval/metrics.json")
    print("Evaluation complete.")
    print(f"Evaluated jobs: {result['evaluated_jobs']}")
    print("Retrieval only:")
    print(
        f"  P@k={result['retrieval_only']['precision_at_k']:.6f} "
        f"R@k={result['retrieval_only']['recall_at_k']:.6f} "
        f"MRR={result['retrieval_only']['mrr']:.6f}"
    )
    print("Retrieval + rerank:")
    print(
        f"  P@k={result['retrieval_plus_rerank']['precision_at_k']:.6f} "
        f"R@k={result['retrieval_plus_rerank']['recall_at_k']:.6f} "
        f"MRR={result['retrieval_plus_rerank']['mrr']:.6f}"
    )


if __name__ == "__main__":
    main()
