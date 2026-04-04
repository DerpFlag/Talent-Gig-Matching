from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.embeddings.chroma_store import upsert_resume_documents
from src.embeddings.retriever import clear_collection_cache, encode_texts_normalized
from src.nlp.skill_extractor import extract_skills


def _normalize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            raise ValueError(f"metadata key {key!r} cannot be None")
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def ingest_resume_entries(
    entries: list[tuple[str, str]],
    extra_metadatas: list[dict[str, Any]] | None = None,
) -> list[str]:
    if not entries:
        raise ValueError("entries cannot be empty")
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))
    chroma_dir = paths["chroma_dir"]
    collection_name = "resumes"
    embedding_model = model_cfg["embedding_model_name"]

    resume_ids = [str(pair[0]) for pair in entries]
    texts = [str(pair[1]) for pair in entries]
    vectors = encode_texts_normalized(embedding_model, texts)

    metadatas: list[dict[str, str | int | float | bool]] = []
    for i, text in enumerate(texts):
        skills = extract_skills(text)
        base_meta: dict[str, Any] = {
            "skills": ",".join(sorted(skills)),
        }
        if extra_metadatas is not None:
            if i >= len(extra_metadatas):
                raise ValueError("extra_metadatas length must match entries when provided")
            base_meta.update(extra_metadatas[i])
        metadatas.append(_normalize_metadata(base_meta))

    upsert_resume_documents(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        resume_ids=resume_ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    clear_collection_cache()
    return resume_ids
