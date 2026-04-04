from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.retriever import retrieve_top_k_resumes
from src.utils.io import read_csv_strict


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))

    jobs_df = read_csv_strict(paths["processed_jobs_path"])
    if "job_text" not in jobs_df.columns:
        raise ValueError("Processed jobs file must contain 'job_text' column.")

    sample_job_text = str(jobs_df.iloc[0]["job_text"])
    results = retrieve_top_k_resumes(
        chroma_dir=paths["chroma_dir"],
        collection_name="resumes",
        embedding_model_name=model_cfg["embedding_model_name"],
        job_text=sample_job_text,
        top_k=int(cfg["retrieval"]["top_k"]),
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    print("Top retrieval results:")
    for idx, (resume_id, distance) in enumerate(zip(ids, distances), start=1):
        print(f"{idx}. resume_id={resume_id} distance={distance:.6f}")


if __name__ == "__main__":
    main()
