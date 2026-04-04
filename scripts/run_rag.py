from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import recommend_candidates
from src.utils.io import read_csv_strict


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    jobs = read_csv_strict(paths["processed_jobs_path"])
    if "job_text" not in jobs.columns:
        raise ValueError("Processed jobs file must include 'job_text'.")

    job_text = str(jobs.iloc[0]["job_text"])
    output = recommend_candidates(job_text=job_text)
    print(f"Job query: {job_text[:120]}...")
    print("Top reranked candidates:")
    for idx, row in enumerate(output["candidates"], start=1):
        matched = ",".join(row["explanation"]["matched_skills"])
        print(
            f"{idx}. resume_id={row['resume_id']} "
            f"score={row['rerank_score']:.6f} "
            f"distance={row['retrieval_distance']:.6f} "
            f"matched_skills={matched}"
        )


if __name__ == "__main__":
    main()
