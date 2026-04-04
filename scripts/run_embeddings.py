from pathlib import Path
import ast
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.chroma_store import build_chroma_resume_index
from src.embeddings.embedder import encode_texts
from src.utils.io import read_csv_strict


def parse_skill_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("skills column must be a serialized list")
    return [str(x) for x in parsed]


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))

    resumes = read_csv_strict(paths["processed_resumes_path"])
    resumes["skills"] = resumes["skills"].apply(parse_skill_list)

    vectors = encode_texts(
        model_name=model_cfg["embedding_model_name"],
        texts=resumes["resume_text"].tolist(),
    )
    ids = resumes["resume_id"].astype(str).tolist()
    docs = resumes["resume_text"].tolist()
    metadata = [{"skills": ",".join(s)} for s in resumes["skills"].tolist()]

    build_chroma_resume_index(
        chroma_dir=paths["chroma_dir"],
        collection_name="resumes",
        resume_ids=ids,
        embeddings=vectors,
        documents=docs,
        metadatas=metadata,
    )
    print("Chroma index created for resumes.")


if __name__ == "__main__":
    main()
