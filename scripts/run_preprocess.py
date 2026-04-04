from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import preprocess_jobs, preprocess_resumes
from src.utils.io import read_csv_strict, ensure_parent


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    resumes = read_csv_strict(paths["raw_resumes_path"])
    jobs = read_csv_strict(paths["raw_jobs_path"])

    resumes_out = preprocess_resumes(resumes)
    jobs_out = preprocess_jobs(jobs)

    ensure_parent(paths["processed_resumes_path"])
    ensure_parent(paths["processed_jobs_path"])
    resumes_out.to_csv(paths["processed_resumes_path"], index=False)
    jobs_out.to_csv(paths["processed_jobs_path"], index=False)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
