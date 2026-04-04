import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def skill_overlap(job_skills: list[str], resume_skills: list[str]) -> float:
    js = set(job_skills)
    rs = set(resume_skills)
    if not js:
        return 0.0
    return len(js & rs) / len(js)


def build_weak_labels(
    jobs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    embedding_model_name: str,
    w_embed: float,
    w_skill: float,
    pos_threshold: float,
    neg_threshold: float,
) -> pd.DataFrame:
    if not (0.0 <= w_embed <= 1.0 and 0.0 <= w_skill <= 1.0):
        raise ValueError("w_embed and w_skill must be between 0 and 1")
    if abs((w_embed + w_skill) - 1.0) > 1e-9:
        raise ValueError("w_embed and w_skill must sum to 1.0")
    model = SentenceTransformer(embedding_model_name)
    job_vecs = model.encode(jobs_df["job_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    resume_vecs = model.encode(resumes_df["resume_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    sim_matrix = np.matmul(job_vecs, resume_vecs.T)

    rows = []
    for i, job in jobs_df.reset_index(drop=True).iterrows():
        for j, resume in resumes_df.reset_index(drop=True).iterrows():
            emb = float(sim_matrix[i, j])
            ov = float(skill_overlap(job["skills"], resume["skills"]))
            score = w_embed * emb + w_skill * ov
            if score >= pos_threshold:
                label = 1
            elif score <= neg_threshold:
                label = 0
            else:
                continue
            rows.append(
                {
                    "job_id": job["job_id"],
                    "resume_id": resume["resume_id"],
                    "embedding_similarity": emb,
                    "skill_overlap": ov,
                    "weak_score": score,
                    "label": label,
                }
            )
    if not rows:
        raise ValueError("No weak labels generated. Adjust thresholds.")
    return pd.DataFrame(rows)


def build_similarity_matrix(
    jobs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    embedding_model_name: str,
) -> np.ndarray:
    model = SentenceTransformer(embedding_model_name)
    job_vecs = model.encode(jobs_df["job_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    resume_vecs = model.encode(resumes_df["resume_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    return np.matmul(job_vecs, resume_vecs.T)
