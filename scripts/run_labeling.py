from pathlib import Path
import ast
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.pair_builder import build_topk_random_pairs
from src.labeling.weak_supervision import build_similarity_matrix
from src.utils.io import read_csv_strict, ensure_parent


def parse_skill_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("skills column must be a serialized list")
    return [str(x) for x in parsed]


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))

    resumes = read_csv_strict(paths["processed_resumes_path"])
    jobs = read_csv_strict(paths["processed_jobs_path"])

    resumes["skills"] = resumes["skills"].apply(parse_skill_list)
    jobs["skills"] = jobs["skills"].apply(parse_skill_list)

    sim_matrix = build_similarity_matrix(
        jobs_df=jobs,
        resumes_df=resumes,
        embedding_model_name=model_cfg["embedding_model_name"],
    )
    labels = build_topk_random_pairs(
        jobs_df=jobs,
        resumes_df=resumes,
        sim_matrix=sim_matrix,
        w_embed=float(cfg["weights"]["embedding_similarity"]),
        w_skill=float(cfg["weights"]["skill_overlap"]),
        positive_top_k=int(cfg["labeling"]["positive_top_k"]),
        negative_random_k=int(cfg["labeling"]["negative_random_k"]),
        seed=int(cfg["random_seed"]),
    )
    ensure_parent(paths["labels_path"])
    labels.to_csv(paths["labels_path"], index=False)
    print(f"Weak labels generated: {len(labels)} rows")


if __name__ == "__main__":
    main()
