from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml

from src.embeddings.retriever import retrieve_top_k_resumes
from src.rag.explain import build_explanation
from src.rag.reranker import get_cached_reranker, rerank_candidates


@lru_cache(maxsize=1)
def _load_paths() -> dict:
    return yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_base_cfg() -> dict:
    return yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_model_cfg() -> dict:
    return yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))


def _extract_retrieval_payload(results: dict) -> tuple[list[str], list[str], list[float]]:
    ids = results.get("ids")
    docs = results.get("documents")
    distances = results.get("distances")
    if not ids or not docs or not distances:
        raise ValueError("Retrieval results are missing required fields.")
    if not ids[0] or not docs[0] or not distances[0]:
        raise ValueError("Retrieval returned empty top-k set.")
    return ids[0], docs[0], distances[0]


def recommend_candidates(job_text: str, top_k: int | None = None) -> dict[str, Any]:
    if not job_text:
        raise ValueError("job_text cannot be empty")

    paths = _load_paths()
    base_cfg = _load_base_cfg()
    model_cfg = _load_model_cfg()

    selected_top_k = int(top_k) if top_k is not None else int(base_cfg["retrieval"]["top_k"])
    retrieval = retrieve_top_k_resumes(
        chroma_dir=paths["chroma_dir"],
        collection_name="resumes",
        embedding_model_name=model_cfg["embedding_model_name"],
        job_text=job_text,
        top_k=selected_top_k,
    )
    candidate_ids, candidate_docs, candidate_distances = _extract_retrieval_payload(retrieval)

    reranker = get_cached_reranker(
        encoder_name=model_cfg["embedding_model_name"],
        model_path=model_cfg["model_output_path"],
        max_length=int(model_cfg["max_length"]),
    )
    ranked = rerank_candidates(
        reranker=reranker,
        job_text=job_text,
        candidate_ids=[str(x) for x in candidate_ids],
        candidate_texts=[str(x) for x in candidate_docs],
        candidate_distances=[float(x) for x in candidate_distances],
    )

    for row in ranked:
        row["explanation"] = build_explanation(job_text=job_text, resume_text=row["resume_text"])
    return {"job_text": job_text, "top_k": selected_top_k, "candidates": ranked}
