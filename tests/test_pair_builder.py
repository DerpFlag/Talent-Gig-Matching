import numpy as np
import pandas as pd

from src.labeling.pair_builder import build_topk_random_pairs


def test_build_topk_random_pairs_shape_and_labels() -> None:
    jobs = pd.DataFrame(
        {
            "job_id": ["j1", "j2"],
            "job_text": ["python nlp", "sql analytics"],
            "skills": [["python", "nlp"], ["sql"]],
        }
    )
    resumes = pd.DataFrame(
        {
            "resume_id": ["r1", "r2", "r3", "r4"],
            "resume_text": ["python nlp", "java", "sql", "python sql"],
            "skills": [["python", "nlp"], ["java"], ["sql"], ["python", "sql"]],
        }
    )
    sim = np.array(
        [
            [0.9, 0.1, 0.2, 0.6],
            [0.2, 0.1, 0.8, 0.7],
        ]
    )
    pairs = build_topk_random_pairs(
        jobs_df=jobs,
        resumes_df=resumes,
        sim_matrix=sim,
        w_embed=0.7,
        w_skill=0.3,
        positive_top_k=1,
        negative_random_k=2,
        seed=42,
    )
    assert len(pairs) == 6
    counts = pairs.groupby("job_id")["label"].value_counts().to_dict()
    assert counts[("j1", 1)] == 1
    assert counts[("j1", 0)] == 2
    assert counts[("j2", 1)] == 1
    assert counts[("j2", 0)] == 2
