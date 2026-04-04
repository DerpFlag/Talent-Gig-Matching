from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_script(script_name: str, timeout_sec: int) -> tuple[int, str]:
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        return 1, f"Missing script: {script_path}"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


LESSONS: list[dict] = [
    {
        "id": 1,
        "title": "Step 1 — Preprocess",
        "script": "run_preprocess.py",
        "timeout": 600,
        "teach": """
**What this does**  
Cleans resume and job text and adds columns like skills, experience years, and simple entity counts.

**Example**  
- Before: `  5 YEARS  in PYTHON `  
- After: `5 years in python`

**Files**  
`scripts/run_preprocess.py` → `src/data/preprocess.py` → writes `data/processed/*.csv`

**You need**  
`data/raw/resumes.csv` and `data/raw/jobs.csv` with columns `resume_id`, `resume_text` and `job_id`, `job_text`.
        """,
    },
    {
        "id": 2,
        "title": "Step 2 — Weak labels (training pairs)",
        "script": "run_labeling.py",
        "timeout": 3600,
        "teach": """
**What this does**  
Builds approximate “match / no-match” pairs for every job using embedding similarity + skill overlap, then **top-k positives** and **random negatives**.

**Example score**  
If job skills are `{python, nlp}` and resume has `{python, sql}`, overlap is part of the score; embeddings add “meaning” similarity.

**Files**  
`scripts/run_labeling.py` → `src/labeling/*` → `data/labels/job_resume_pairs.csv`

**Settings**  
See `configs/base.yaml` → `labeling.positive_top_k`, `labeling.negative_random_k`.
        """,
    },
    {
        "id": 3,
        "title": "Step 3 — Embeddings + Chroma index",
        "script": "run_embeddings.py",
        "timeout": 3600,
        "teach": """
**What this does**  
Turns each resume into a vector (embedding) and stores it in **ChromaDB** for fast similarity search.

**Why**  
At query time we cannot compare the job to every resume with a heavy model on the full corpus; the vector DB finds likely candidates quickly.

**Output**  
`data/artifacts/chroma`
        """,
    },
    {
        "id": 4,
        "title": "Step 4 — Train reranker",
        "script": "run_train.py",
        "timeout": 7200,
        "teach": """
**What this does**  
Trains a **Siamese-style** model (PyTorch) to score job–resume pairs so we can **rerank** the retrieved list.

**Output**  
`data/artifacts/model/best_model.pt`

**Note**  
This step can take a long time on CPU. Be patient or use a smaller dataset first.
        """,
    },
    {
        "id": 5,
        "title": "Step 5 — Try retrieval only",
        "script": "run_retrieval.py",
        "timeout": 600,
        "teach": """
**What this does**  
Runs semantic search only (no reranker). Useful to see “raw” vector retrieval ordering.

**File**  
`scripts/run_retrieval.py`
        """,
    },
    {
        "id": 6,
        "title": "Step 6 — Full RAG path (retrieve + rerank)",
        "script": "run_rag.py",
        "timeout": 1200,
        "teach": """
**What this does**  
End-to-end: retrieve top-k from Chroma, then rerank with the trained model, then add skill explanations.

**This is the same idea as the live API**, but from the command-line script.
        """,
    },
    {
        "id": 7,
        "title": "Step 7 — Evaluation metrics",
        "script": "run_eval.py",
        "timeout": 3600,
        "teach": """
**What this does**  
Computes Precision@k, Recall@k, MRR for retrieval-only vs retrieval+rerank.

**Output**  
`data/artifacts/eval/metrics.json`
        """,
    },
    {
        "id": 8,
        "title": "Optional — Latency benchmark",
        "script": None,
        "timeout": 3600,
        "teach": """
**What this does**  
Measures how fast recommendations run (warm-up excluded). Uses CLI args.

**Example command** (run in terminal if you prefer):  
`python scripts/run_benchmark.py --sample-size 50 --warmup 5 --top-k 10`

Below you can run with the parameters you choose.
        """,
    },
]
