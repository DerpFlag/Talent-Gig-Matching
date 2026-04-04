from src.rag.pipeline import recommend_candidates


def run_recommendation(job_description: str, top_k: int | None) -> dict:
    if not job_description or not isinstance(job_description, str):
        raise ValueError("job_description must be a non-empty string")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0 when provided")
    return recommend_candidates(job_text=job_description, top_k=top_k)
