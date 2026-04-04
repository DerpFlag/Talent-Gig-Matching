---
title: Talent-Gig Matcher
emoji: briefcase
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Talent-Gig Matching System

Production-style local-first project to match candidate resumes to gig/job descriptions using NLP, embeddings, vector retrieval, and reranking.

**Hugging Face Space:** Streamlit is **not** a separate SDK on Spaces anymore. This repo uses **`sdk: docker`** and the root **`Dockerfile`** runs the Streamlit product app on port **7860**. On [Create new Space](https://huggingface.co/new-space) choose **Docker** (not Gradio), then import [the GitHub repo](https://github.com/DerpFlag/Talent-Gig-Matching). See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Phases
- Phase 1: Problem framing and system design
- Phase 2: Data preprocessing and weak-label generation
- Phase 3: Embeddings and ChromaDB retrieval
- Phase 4: Model training and reranking
- Phase 5: Retrieval + RAG pipeline
- Phase 6: Evaluation (Precision@k, Recall@k, MRR)
- Phase 7: Deployment (FastAPI + UI)

## Local Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Set environment variables:
   - `KAGGLE_API_TOKEN`
   - `HF_TOKEN`
4. Ensure dataset CSV files are placed in:
   - `data/raw/resumes.csv`
   - `data/raw/jobs.csv`

## No-Fallback Policy
This repository is configured to fail explicitly if any required file/config/service is missing.

## Initial Commands
- Phase 2 preprocessing: `python scripts/run_preprocess.py`
- Phase 2 weak labels (top-k positives + random negatives): `python scripts/run_labeling.py`
- Phase 3 embeddings/chroma: `python scripts/run_embeddings.py`
- Phase 3 retrieval test: `python scripts/run_retrieval.py`
- Phase 4 training: `python scripts/run_train.py`
- Phase 5 RAG retrieval + rerank: `python scripts/run_rag.py`
- Phase 6 evaluation: `python scripts/run_eval.py`
- Phase 7 API server: `python scripts/run_api.py`
- Phase 7 Streamlit product app (PDF ingest, match, docs, advanced pipeline): `python scripts/run_ui.py`
- Teacher UI (learn-by-clicking + guide + Try API): `python scripts/run_teacher_ui.py`
- Minimal Streamlit demo (API only, port 8502): `python scripts/run_ui_demo.py`
- API additions: `POST /ingest/resume-pdfs` (multipart PDFs), `POST /ingest/resume-texts` (JSON batch)
- Benchmark report: `python scripts/run_benchmark.py`
- Benchmark (custom): `python scripts/run_benchmark.py --sample-size 50 --warmup 5 --top-k 10 --output data/artifacts/reports/benchmark_custom.json`

## API Contract
### POST `/recommend`
Request JSON:
```json
{
  "job_description": "Looking for an NLP engineer with PyTorch and FastAPI experience",
  "top_k": 25
}
```

Response JSON:
- `job_text`
- `top_k`
- `candidates[]` with `resume_id`, `retrieval_distance`, `rerank_score`, `resume_text`, and `explanation`

## Deployment Notes
- Local API: FastAPI + Uvicorn
- Local UI: Streamlit
- Cloud API option: Render/Railway (Docker or Python web service)
- Cloud UI option: Streamlit Community Cloud
- Docker Compose local stack: `docker compose up --build` (API/UI images use `Dockerfile.api`; HF Space uses root `Dockerfile` + port 7860)

## Quality Gates
- Run tests: `python -m pytest -q`
- CI workflow: `.github/workflows/ci.yml`
- Docker image builds:
  - Streamlit (HF / product UI): `docker build -t talentgig-ui-space .`
  - API (local compose): `docker build -f Dockerfile.api -t talentgig-api .`

## Portfolio Artifacts
- Model card: `docs/MODEL_CARD.md`
- Experiment template: `docs/EXPERIMENT_REPORT.md`
- Non-technical project guide: `docs/NON_TECH_PROJECT_GUIDE.md`
- Deploy notes (GitHub vs Hugging Face Spaces, SaaS reality check): `docs/DEPLOYMENT.md`
- Benchmark output: `data/artifacts/reports/benchmark.json`
- One-click setup:
  - Windows: `powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1`
  - Linux/macOS: `bash scripts/bootstrap.sh`

## Benchmark Results
Latest local benchmark from `scripts/run_benchmark.py`:

| Metric | Value |
|---|---:|
| Total sampled queries | 8 |
| Warm-up queries (excluded) | 5 |
| Measured queries | 3 |
| Avg latency | 558.14 ms |
| P50 latency | 560.94 ms |
| P90 latency | 566.21 ms |
| P95 latency | 566.87 ms |
| P99 latency | 567.40 ms |
| Min latency | 545.96 ms |
| Max latency | 567.53 ms |
| Trimmed avg (10%) | 558.14 ms |

Notes:
- Measured on local CPU with retrieval + rerank end-to-end path.
- Inference caching optimization is enabled (embedder + reranker loaded once per process).
- Warm-up requests are excluded, so values represent steady-state behavior.
- Current processed job dataset size is small; increase dataset size for more statistically stable benchmark distributions.

CLI options for benchmark:
- `--sample-size`: number of processed jobs to benchmark
- `--warmup`: number of warm-up queries excluded from metrics
- `--top-k`: retrieval/re-rank candidate count
- `--output`: output JSON report path
