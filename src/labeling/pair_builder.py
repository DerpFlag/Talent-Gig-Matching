import random
import numpy as np
import pandas as pd

from src.labeling.weak_supervision import skill_overlap


def _build_score_frame(
    jobs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    w_embed: float,
    w_skill: float,
) -> pd.DataFrame:
    rows = []
    jobs = jobs_df.reset_index(drop=True)
    resumes = resumes_df.reset_index(drop=True)
    for i, job in jobs.iterrows():
        for j, resume in resumes.iterrows():
            emb = float(sim_matrix[i, j])
            ov = float(skill_overlap(job["skills"], resume["skills"]))
            score = w_embed * emb + w_skill * ov
            rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "resume_id": str(resume["resume_id"]),
                    "embedding_similarity": emb,
                    "skill_overlap": ov,
                    "weak_score": score,
                }
            )
    return pd.DataFrame(rows)


def build_topk_random_pairs(
    jobs_df: pd.DataFrame,
    resumes_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    w_embed: float,
    w_skill: float,
    positive_top_k: int,
    negative_random_k: int,
    seed: int,
) -> pd.DataFrame:
    if positive_top_k <= 0 or negative_random_k <= 0:
        raise ValueError("positive_top_k and negative_random_k must be > 0")
    random.seed(seed)
    scored = _build_score_frame(jobs_df, resumes_df, sim_matrix, w_embed, w_skill)
    grouped = scored.groupby("job_id", sort=False)
    all_rows = []
    for _, grp in grouped:
        grp_sorted = grp.sort_values("weak_score", ascending=False).reset_index(drop=True)
        positives = grp_sorted.head(positive_top_k).copy()
        positives["label"] = 1

        remaining = grp_sorted.iloc[positive_top_k:].copy()
        if len(remaining) < negative_random_k:
            raise ValueError("Insufficient remaining candidates for random negative sampling.")
        negatives = remaining.sample(n=negative_random_k, random_state=seed).copy()
        negatives["label"] = 0
        all_rows.append(pd.concat([positives, negatives], axis=0))

    out = pd.concat(all_rows, axis=0).reset_index(drop=True)
    if out.empty:
        raise ValueError("No training pairs generated.")
    return out
