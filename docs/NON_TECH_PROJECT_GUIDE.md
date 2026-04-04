# Talent-Gig Matching System: Complete Non-Technical Master Guide

This document is written for a non-technical audience.
It explains, in detail:
- what we built,
- why we built it,
- how each part works,
- what every key file does,
- what each technical word means,
- what results we got,
- and how to interpret those results.

If you read this from top to bottom, you will understand the project end-to-end without needing coding experience.

---

## Part A: The Business Story

## A1) The core problem in simple words

When companies hire for gigs/jobs, they usually get many resumes.
The hard part is finding the right people quickly.

This is difficult because:
- Resumes are written in many different styles.
- Job descriptions are often unclear or inconsistent.
- The same skill can be written in different ways ("ML", "machine learning", "AI modeling").
- Manual screening is slow and expensive.

So the business pain is:
- too much recruiter time,
- slower hiring,
- missed quality candidates,
- lower confidence in shortlist quality.

## A2) What this project does

This project acts as a smart recommendation assistant:
1. You give it a job description.
2. It searches a large resume pool by meaning (not just exact words).
3. It ranks candidates from best to least relevant.
4. It gives simple explanation clues (matched/missing skills).

Think of it as:
- "Google-like search for resumes" + "AI ranking layer for quality ordering."

Example:
- Input job: "Need Python, NLP, FastAPI, and PyTorch experience."
- System output (simplified):
  1) Resume A (best match)
  2) Resume B
  3) Resume C
- Explanation snippet:
  - matched skills: `python`, `nlp`
  - missing requested skills: `fastapi`, `pytorch`

## A3) Who benefits

- Recruiters: less manual filtering.
- Hiring managers: better shortlist quality.
- Gig platforms: better matching efficiency.
- Candidates: more relevant matching opportunities.

## A4) What success means

We track these quality metrics:
- **Precision@k**: among top `k` recommendations, how many are correct/relevant.
- **Recall@k**: out of all relevant candidates, how many were found in top `k`.
- **MRR**: how early the first relevant candidate appears.
- **Latency**: how fast the system responds.

Example with `k = 5`:
- If top 5 results contain 4 truly relevant candidates, Precision@5 = `4/5 = 0.80`.
- If there are 10 relevant candidates total in the full dataset and top 5 contains 4 of them, Recall@5 = `4/10 = 0.40`.
- If first relevant candidate appears at rank 2, reciprocal rank = `1/2 = 0.5`. MRR averages this across many queries.

Business meaning:
- Higher precision: better shortlist quality.
- Higher recall: fewer strong candidates missed.
- Higher MRR: good candidates appear earlier.
- Lower latency: faster recruiter workflow.

---

## Part B: Data Story (Where data came from and why)

## B1) Data sources used

We used public Kaggle datasets:

1) Resume data:
- Kaggle source: `snehaanbhawal/resume-dataset`
- Raw extracted file:
  - `data/raw/kaggle/resume-dataset/Resume/Resume.csv`

2) Job data:
- Kaggle source: `andrewmvd/data-scientist-jobs`
- Raw extracted file:
  - `data/raw/kaggle/data-scientist-jobs/DataScientist.csv`

## B2) Why we had to transform raw data

Different datasets use different column names and formats.
Our pipeline needs one clean standard format.

So we normalized into:
- `data/raw/resumes.csv` with:
  - `resume_id`, `resume_text`
- `data/raw/jobs.csv` with:
  - `job_id`, `job_text`

Example:
- Original resume column in source might be `Resume_str`.
- We rename/map it into `resume_text`.
- Original job column might be `Job Description`.
- We rename/map it into `job_text`.

## B3) The hardest data issue

There is no perfect public dataset saying:
"This resume is definitely a match for this job."

So we created training labels ourselves using weak supervision.
This is common in real-world ML projects when labeled data is limited.

---

## Part C: Technical Terms (explained simply)

Below are core terms and what they mean in this project.

- **NLP**: teaching computers to understand human language text.
- **Transformer**: a modern AI architecture used for language understanding.
- **Embedding**: converting text into numeric vectors (math-friendly meaning representation).
- **Semantic search**: finding results by meaning, not exact keyword matching.
- **Vector DB**: database optimized for embedding similarity search.
- **ChromaDB**: the vector database used here.
- **Weak supervision**: generating approximate training labels instead of manually labeling everything.
- **Top-k**: top `k` results (for example top 10).
- **Reranking**: reordering initially retrieved results using a stronger model.
- **Siamese model**: compares two texts (job + resume) and predicts match score.
- **RAG-style pipeline (in this project)**: retrieve candidates first, then rerank.
- **API**: interface other apps can call programmatically.
- **FastAPI**: API framework used here.
- **Streamlit**: simple web app framework used for demo UI.
- **CI (Continuous Integration)**: automatic tests run when code changes.
- **Docker**: package app + dependencies into repeatable container.

Concrete mini-examples:
- Embedding example:
  - text A: "Python NLP engineer"
  - text B: "Natural language engineer with Python"
  - These get embeddings that are close in vector space, so the system sees them as similar.
- Top-k example:
  - If `k = 10`, system returns top 10 most relevant candidates first.
- Reranking example:
  - Retrieval returns candidates `[R1, R2, R3]`.
  - Reranker rescoring changes order to `[R2, R1, R3]` because R2 is judged more relevant after deeper comparison.

---

## Part D: End-to-End Process (Step by Step)

This is the exact lifecycle of data and model behavior in this project.

## D1) Step 1 - Preprocess and enrich text

Entrypoint:
- `scripts/run_preprocess.py`

Core modules:
- `src/data/preprocess.py`
- `src/nlp/normalizer.py`
- `src/nlp/skill_extractor.py`
- `src/nlp/entity_parser.py`
- `src/nlp/experience_parser.py`

What happens:
1. Text normalization (cleaning, lowercasing, spacing normalization).
2. Skills extraction (rule-based skill detection).
3. Entity extraction (email/phone/url counts).
4. Experience extraction (years from text patterns).

Examples:
1. Text normalization
   - Before: `"  5 YEARS  in PYTHON & NLP.  "`
   - After: `"5 years in python & nlp."`
2. Skills extraction
   - Input text: `"Worked with Python, FastAPI, and Docker"`
   - Output skills list: `["python", "fastapi", "docker"]`
3. Entity extraction
   - Input contains `john@abc.com` and `+1 555-123-4567`
   - Output counts: `email_count=1`, `phone_count=1`
4. Experience extraction
   - Input: `"7 years of data science experience"`
   - Extracted value: `7`

Outputs:
- `data/processed/resumes_processed.csv`
- `data/processed/jobs_processed.csv`

Why this matters:
- Better input quality -> better matching quality.
- Adds structured fields helpful for analysis/explanation.

## D2) Step 2 - Build weak-labeled training pairs

Entrypoint:
- `scripts/run_labeling.py`

Core modules:
- `src/labeling/weak_supervision.py`
- `src/labeling/pair_builder.py`
- `configs/base.yaml`

What happens:
1. Compute embedding similarity between each job and each resume.
2. Compute skill overlap for each pair.
3. Combine both signals into a hybrid score.
4. For each job:
   - top-k highest scores => positive examples
   - random sampled non-top examples => negative examples

Example pair scoring:
- Job skills: `{python, nlp, fastapi}`
- Resume skills: `{python, nlp, sql}`
- Skill overlap = `2/3 = 0.67`
- Embedding similarity (example): `0.82`
- If weights are `w1=0.7` and `w2=0.3`:
  - final score = `0.7*0.82 + 0.3*0.67 = 0.775`
- If this score is in top-k for that job, this pair becomes a positive example.

Output:
- `data/labels/job_resume_pairs.csv`

Why this matters:
- Creates a practical training dataset when true labels are unavailable.
- Gives enough signal to train a ranking model.

## D3) Step 3 - Build semantic retrieval index

Entrypoint:
- `scripts/run_embeddings.py`

Core modules:
- `src/embeddings/embedder.py`
- `src/embeddings/chroma_store.py`

What happens:
1. Convert resume text into embeddings.
2. Store vectors + metadata into ChromaDB.

Example:
- Resume text: "Built NLP APIs using Python and FastAPI."
- Stored vector: `[0.12, -0.33, 0.04, ...]` (many dimensions)
- Stored metadata: extracted skills, IDs, and original text snippet.

Output:
- vector index persisted under `data/artifacts/chroma`

Why this matters:
- Fast candidate retrieval at runtime.
- Needed for scalable search before reranking.

## D4) Step 4 - Train reranking model

Entrypoint:
- `scripts/run_train.py`

Core modules:
- `src/models/dataset.py`
- `src/models/siamese_model.py`
- `src/models/losses.py`
- `src/models/train.py`
- `configs/model.yaml`

What happens:
1. Build text-pair dataset from weak labels.
2. Tokenize job/resume text for transformer input.
3. Train Siamese-style model to predict match likelihood.
4. Save best model artifact.

Example:
- Training input pair:
  - Job: "Need NLP engineer with PyTorch"
  - Resume: "Built text models using PyTorch and transformers"
  - Label: `1` (match)
- Another pair with lower relevance gets label `0`.
- Model learns patterns that separate good vs bad matches.

Outputs:
- `data/artifacts/model/best_model.pt`
- `data/artifacts/model/best_model.metrics.json`

Why this matters:
- Retrieval gives fast candidates.
- This model improves ranking quality among retrieved candidates.

## D5) Step 5 - Retrieval + Rerank pipeline (RAG-style)

Entrypoint:
- `scripts/run_rag.py`

Core modules:
- `src/rag/pipeline.py`
- `src/embeddings/retriever.py`
- `src/rag/reranker.py`
- `src/rag/explain.py`

What happens at inference:
1. Job description -> embedding.
2. ChromaDB retrieves top candidates by vector similarity.
3. Reranker model rescoring improves order.
4. Explanation generator adds matched/missing skills.

Worked flow example:
1. Job query: "Need Python + NLP + FastAPI"
2. Retrieval returns 10 resumes quickly by semantic similarity.
3. Reranker inspects those 10 in more detail and reorders them.
4. Final output for one candidate:
   - rerank_score: `0.87`
   - matched_skills: `python, nlp`
   - missing_job_skills: `fastapi`

Why this design:
- Retrieve stage = speed.
- Rerank stage = precision.
- Combined = practical latency + quality.

## D6) Step 6 - Evaluate quality

Entrypoint:
- `scripts/run_eval.py`

Core modules:
- `src/eval/metrics.py`
- `src/eval/evaluate.py`

What happens:
- Compare retrieval-only vs retrieval+rerank.
- Compute Precision@k, Recall@k, MRR.

Example interpretation:
- Retrieval-only Precision@10 = 0.50
- Retrieval+Rerank Precision@10 = 0.70
- Meaning: after reranking, top 10 shortlist quality improved.

Output:
- `data/artifacts/eval/metrics.json`

Why this matters:
- Provides objective evidence of ranking behavior.
- Supports portfolio and business discussion.

## D7) Step 7 - Deploy as usable product

API:
- `scripts/run_api.py`
- `src/api/main.py`
- `src/api/schemas.py`
- `src/api/service.py`

UI:
- `scripts/run_ui.py`
- `src/ui/app.py`

What happens:
- API endpoint `/recommend` receives job text and returns ranked candidates.
- Streamlit UI provides a browser interface for non-technical usage.

API example request:
```json
{
  "job_description": "Looking for a Python NLP engineer with FastAPI experience",
  "top_k": 5
}
```

API example response (simplified):
```json
{
  "job_text": "...",
  "top_k": 5,
  "candidates": [
    {
      "resume_id": "29149998",
      "rerank_score": 0.45,
      "explanation": {
        "matched_skills": ["python", "nlp"],
        "missing_job_skills": ["fastapi"]
      }
    }
  ]
}
```

Why this matters:
- Converts ML pipeline into an actual usable product.

---

## Part E: Full File-by-File Purpose Map

This section explains each main file specifically.

## E1) Root-level platform files

- `README.md`: project overview, run commands, benchmark summary.
- `requirements.txt`: Python dependencies list.
- `Dockerfile`: container image for API runtime.
- `docker-compose.yml`: local multi-service stack (API + UI).
- `.dockerignore`: excludes unnecessary files from Docker build context.
- `progress.md`: project milestone log.
- `apis.txt`: local token registry (sensitive local file).
- `kaggle.json`: Kaggle auth credentials file (local secret).

## E2) Config files

- `configs/base.yaml`: core project settings (weights, thresholds, retrieval size, labeling sampling rules).
- `configs/paths.yaml`: canonical file locations for datasets/artifacts.
- `configs/model.yaml`: model and training hyperparameters.

Example settings:
- In `configs/base.yaml`:
  - `retrieval.top_k: 25` means retrieve 25 candidates before reranking.
- In `configs/model.yaml`:
  - `max_length: 64` controls how much text is fed to model at once.

## E3) Script entrypoints (`scripts/`)

- `run_preprocess.py`: run preprocessing and feature extraction.
- `run_labeling.py`: generate weak-labeled job-resume pairs.
- `run_embeddings.py`: build Chroma embedding index.
- `run_retrieval.py`: test retrieval-only behavior.
- `run_train.py`: train Siamese reranker.
- `run_rag.py`: run full retrieve+rerank pipeline.
- `run_eval.py`: compute ranking metrics.
- `run_api.py`: start FastAPI service.
- `run_ui.py`: start Streamlit app.
- `run_benchmark.py`: benchmark inference latency with configurable settings.
- `bootstrap.ps1`: Windows one-click setup.
- `bootstrap.sh`: Linux/macOS one-click setup.

Example command usage:
- `python scripts/run_benchmark.py --sample-size 50 --warmup 5 --top-k 10`
  - tests 50 jobs,
  - ignores first 5 (warm-up),
  - measures steady-state latency for top-10 retrieval/rerank.

## E4) Source code modules (`src/`)

### `src/data/`
- `preprocess.py`: applies normalization + skill/entity/experience extraction and writes processed tables.

### `src/nlp/`
- `normalizer.py`: text cleanup helper.
- `skill_extractor.py`: skill keyword extraction logic.
- `entity_parser.py`: counts emails/phones/urls.
- `experience_parser.py`: extracts years of experience.

### `src/labeling/`
- `weak_supervision.py`: similarity matrix and weak score functions.
- `pair_builder.py`: top-k positive + random negative pair constructor.

### `src/embeddings/`
- `embedder.py`: embedding generation.
- `chroma_store.py`: write vectors to ChromaDB.
- `retriever.py`: query ChromaDB by embedding (with runtime caching).

### `src/models/`
- `dataset.py`: model input dataset/tokenization.
- `siamese_model.py`: core pair-scoring neural network.
- `losses.py`: weighted BCE loss config.
- `train.py`: complete train/validation pipeline and artifact saving.

### `src/rag/`
- `pipeline.py`: orchestration of retrieve + rerank + explanation.
- `reranker.py`: model-based rescoring module (cached loader).
- `explain.py`: explanation snippets (skill overlap/missing skill cues).

### `src/eval/`
- `metrics.py`: precision/recall/MRR formulas.
- `evaluate.py`: evaluation orchestration over dataset.

### `src/api/`
- `schemas.py`: request/response contract definitions.
- `service.py`: recommendation service wrapper.
- `main.py`: FastAPI app and endpoints.

### `src/ui/`
- `app.py`: Streamlit browser interface.

### `src/utils/`
- `io.py`: strict file read/validation helpers.

## E5) Testing and quality

- `tests/test_metrics.py`: validates metric formulas.
- `tests/test_pair_builder.py`: validates pair sampling rules.
- `tests/test_api.py`: validates API health + schema validation behavior.
- `.github/workflows/ci.yml`: runs tests automatically in GitHub Actions.

## E6) Documentation artifacts

- `docs/MODEL_CARD.md`: model purpose, limits, responsible-use notes.
- `docs/EXPERIMENT_REPORT.md`: structured template for experiment tracking.
- `docs/NON_TECH_PROJECT_GUIDE.md`: this complete non-technical handbook.

---

## Part F: Tools Used and Why (Practical Explanation)

- **Python**: the language for all data and ML stages.
- **Transformers/SentenceTransformers**: understand text meaning and create embeddings.
- **PyTorch**: train the ranking model (this project uses PyTorch, not TensorFlow/Keras).
- **ChromaDB**: store and search embeddings quickly.
- **FastAPI**: expose model as API endpoint.
- **Streamlit**: easy web demo for non-technical users.
- **PyTest**: automatic checks for logic correctness.
- **Docker + Compose**: make deployment consistent and repeatable.
- **GitHub Actions CI**: ensures tests run when code changes.

Example in everyday terms:
- You push new code.
- CI automatically runs tests.
- If tests fail, you know something broke before deployment.

---

## Part G: Results and Interpretation

## G1) Functional result

The project successfully does:
1. ingest raw datasets,
2. preprocess text,
3. create weak labels,
4. index embeddings,
5. train reranker,
6. run retrieval+rereank,
7. evaluate metrics,
8. serve via API and UI.

## G2) Performance result

Benchmark file:
- `data/artifacts/reports/benchmark.json`

Current benchmark process includes warm-up exclusion and percentile reporting.
Meaning:
- reported values represent steady-state behavior better than cold-start-only timing.

Example:
- First request after startup might be slow due to model loading.
- Warm-up excludes those startup-only costs.
- Reported p95 then better reflects normal day-to-day usage.

## G3) Why this is meaningful

This is not a toy script.
It demonstrates complete ML product workflow:
- data pipeline,
- model training,
- retrieval architecture,
- evaluation,
- deployment,
- testing,
- documentation.

---

## Part H: Known Limitations (Honest, important)

- Weak labels are approximate, not human truth labels.
- Rule-based skills extraction can miss synonyms/domain terms.
- Dataset domain may bias behavior.
- Small benchmark samples are less statistically stable.

Simple example of a limitation:
- Job asks for "LLM orchestration".
- Resume says "prompt pipeline engineering".
- If that phrase is not in the skill dictionary, overlap score may undercount relevance.

These are normal for portfolio-grade v1 systems and can be improved in future iterations.

---

## Part I: How to Run It (Non-technical checklist)

1) Environment setup:
- Windows: `powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1`
- Linux/macOS: `bash scripts/bootstrap.sh`

2) Place data:
- `data/raw/resumes.csv`
- `data/raw/jobs.csv`

3) Run pipeline:
- `python scripts/run_preprocess.py`
- `python scripts/run_labeling.py`
- `python scripts/run_embeddings.py`
- `python scripts/run_train.py`
- `python scripts/run_eval.py`

What each command does (quick examples):
- `run_preprocess.py`: creates cleaned files with extracted skill/experience columns.
- `run_labeling.py`: creates match/no-match training pairs per job.
- `run_embeddings.py`: fills vector DB for semantic search.
- `run_train.py`: creates trained model file (`best_model.pt`).
- `run_eval.py`: produces ranking quality report.

4) Start product:
- API: `python scripts/run_api.py`
- UI: `python scripts/run_ui.py`

5) Optional benchmark:
- `python scripts/run_benchmark.py --sample-size 50 --warmup 5 --top-k 10`

---

## Part J: One-line memory aid

This project is a complete "smart hiring recommender":
**understand text -> retrieve likely resumes -> rerank for quality -> explain result -> serve via API/UI**.

