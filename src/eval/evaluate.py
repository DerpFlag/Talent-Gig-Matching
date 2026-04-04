from pathlib import Path
import json

import pandas as pd
import yaml

from src.embeddings.retriever import retrieve_top_k_resumes
from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.rag.pipeline import _extract_retrieval_payload
from src.rag.reranker import PairReranker, rerank_candidates
from src.utils.io import read_csv_strict


def _build_relevance(labels_df: pd.DataFrame) -> dict[str, set[str]]:
    rel = {}
    for job_id, group in labels_df.groupby("job_id"):
        positives = set(group[group["label"] == 1]["resume_id"].astype(str).tolist())
        rel[str(job_id)] = positives
    return rel


def evaluate_retrieval_and_rerank(output_path: str) -> dict:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))

    labels = read_csv_strict(paths["labels_path"])
    jobs = read_csv_strict(paths["processed_jobs_path"])
    required_labels = {"job_id", "resume_id", "label"}
    required_jobs = {"job_id", "job_text"}
    if required_labels - set(labels.columns):
        raise ValueError(f"Missing label columns: {required_labels - set(labels.columns)}")
    if required_jobs - set(jobs.columns):
        raise ValueError(f"Missing jobs columns: {required_jobs - set(jobs.columns)}")

    top_k = int(base_cfg["retrieval"]["top_k"])
    relevance = _build_relevance(labels)
    eval_jobs = jobs[jobs["job_id"].astype(str).isin(relevance.keys())].copy()
    if eval_jobs.empty:
        raise ValueError("No overlapping jobs between processed jobs and labels.")

    reranker = PairReranker(
        encoder_name=model_cfg["embedding_model_name"],
        model_path=model_cfg["model_output_path"],
        max_length=int(model_cfg["max_length"]),
    )

    retrieval_metrics = {"p": [], "r": [], "rr": []}
    rerank_metrics = {"p": [], "r": [], "rr": []}

    for _, row in eval_jobs.iterrows():
        job_id = str(row["job_id"])
        job_text = str(row["job_text"])
        relevant_ids = relevance[job_id]
        if not relevant_ids:
            continue

        raw = retrieve_top_k_resumes(
            chroma_dir=paths["chroma_dir"],
            collection_name="resumes",
            embedding_model_name=model_cfg["embedding_model_name"],
            job_text=job_text,
            top_k=top_k,
        )
        ids, docs, distances = _extract_retrieval_payload(raw)
        retrieval_ranked_ids = [str(x) for x in ids]

        reranked = rerank_candidates(
            reranker=reranker,
            job_text=job_text,
            candidate_ids=[str(x) for x in ids],
            candidate_texts=[str(x) for x in docs],
            candidate_distances=[float(x) for x in distances],
        )
        rerank_ranked_ids = [str(x["resume_id"]) for x in reranked]

        retrieval_metrics["p"].append(precision_at_k(retrieval_ranked_ids, relevant_ids, top_k))
        retrieval_metrics["r"].append(recall_at_k(retrieval_ranked_ids, relevant_ids, top_k))
        retrieval_metrics["rr"].append(reciprocal_rank(retrieval_ranked_ids, relevant_ids))

        rerank_metrics["p"].append(precision_at_k(rerank_ranked_ids, relevant_ids, top_k))
        rerank_metrics["r"].append(recall_at_k(rerank_ranked_ids, relevant_ids, top_k))
        rerank_metrics["rr"].append(reciprocal_rank(rerank_ranked_ids, relevant_ids))

    if not retrieval_metrics["p"] or not rerank_metrics["p"]:
        raise ValueError("No evaluable jobs found with positive labels.")

    result = {
        "top_k": top_k,
        "retrieval_only": {
            "precision_at_k": float(sum(retrieval_metrics["p"]) / len(retrieval_metrics["p"])),
            "recall_at_k": float(sum(retrieval_metrics["r"]) / len(retrieval_metrics["r"])),
            "mrr": float(sum(retrieval_metrics["rr"]) / len(retrieval_metrics["rr"])),
        },
        "retrieval_plus_rerank": {
            "precision_at_k": float(sum(rerank_metrics["p"]) / len(rerank_metrics["p"])),
            "recall_at_k": float(sum(rerank_metrics["r"]) / len(rerank_metrics["r"])),
            "mrr": float(sum(rerank_metrics["rr"]) / len(rerank_metrics["rr"])),
        },
        "evaluated_jobs": len(retrieval_metrics["p"]),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
