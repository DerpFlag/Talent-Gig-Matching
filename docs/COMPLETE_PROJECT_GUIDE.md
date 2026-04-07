# Complete project guide — single source of truth

**Read this file first.** It ties together everything in the repo: data, weak labels, embeddings, the trainable model, training hyperparameters, inference (RAG-style path), evaluation, API, UI, and deployment.  

**Other files in `docs/`** are shorter slices (templates, deployment URLs, or a long non-technical narrative). They are **not** required if you work from this guide; each is summarized [below](#other-documents-in-docs).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)  
2. [How the pieces connect (diagram in words)](#2-how-the-pieces-connect-diagram-in-words)  
3. [Glossary](#3-glossary)  
4. [Configuration files (single list)](#4-configuration-files-single-list)  
5. [Data: from raw CSV to training rows](#5-data-from-raw-csv-to-training-rows)  
6. [The trainable model (what “model” means here)](#6-the-trainable-model-what-model-means-here)  
7. [Training: loss, optimizer, schedule, split, outputs](#7-training-loss-optimizer-schedule-split-outputs)  
8. [Embeddings and Chroma (retrieval index)](#8-embeddings-and-chroma-retrieval-index)  
9. [Inference: retrieve → rerank → explain](#9-inference-retrieve--rerank--explain)  
10. [Evaluation metrics](#10-evaluation-metrics)  
11. [API, UI, scripts](#11-api-ui-scripts)  
12. [Deployment and production notes](#12-deployment-and-production-notes)  
13. [Stack: PyTorch vs TensorFlow](#13-stack-pytorch-vs-tensorflow)  
14. [Other documents in `docs/`](#other-documents-in-docs)  

---

## 1. What this project is

An end-to-end **talent ↔ gig matching** system:

- **Offline:** Clean résumés/jobs → build **weak-supervised** (job, résumé) pairs → embed résumés into **ChromaDB** → **train** a **Siamese-style matcher** in **PyTorch** (encoder from **Hugging Face Transformers**).  
- **Online:** Embed the job text → **retrieve** top‑k résumés from Chroma → **rerank** with the trained matcher → **explain** overlap using a simple skill dictionary.

There is **no TensorFlow/Keras** in this repository.

---

## 2. How the pieces connect (diagram in words)

```
data/raw/*.csv
    → preprocess (run_preprocess.py) → data/processed/*.csv
    → labeling (run_labeling.py)     → data/labels/job_resume_pairs.csv
    → embeddings (run_embeddings.py) → data/artifacts/chroma/
    → train (run_train.py)           → data/artifacts/model/best_model.pt (+ .metrics.json)
    → RAG path / API / Streamlit     → recommend + optional PDF upsert into Chroma
```

Scripts live under `scripts/`; core logic under `src/`.

---

## 3. Glossary

| Term | Meaning in this project |
|------|-------------------------|
| **Embedding** | Fixed-size vector representing a piece of text; similar texts → closer vectors (with the chosen model). |
| **Encoder** | The pretrained transformer backbone (`AutoModel`) that turns token IDs into contextual vectors; we mean-pool to one vector per text. |
| **Siamese network** | Same encoder applied to **job** and **résumé**; outputs are combined and fed to a **classifier head** to predict match quality. |
| **Weak supervision** | Training labels **not** from human annotators; here, **heuristic scores** (embedding similarity + skill overlap) then **top‑k positives / random negatives** per job. |
| **RAG-style** | **R**etrieval **A**ugmented pattern: **retrieve** candidates cheaply, then **refine** (rerank). We do **not** use a generative LLM in the core path. |
| **ChromaDB** | On-disk **vector store**; stores résumé embeddings + text for approximate nearest-neighbor search. |
| **Reranker** | The trained `SiameseMatcher`; scores each (job, résumé) pair after retrieval. |

---

## 4. Configuration files (single list)

| File | Purpose |
|------|---------|
| `configs/paths.yaml` | All important paths: raw/processed CSVs, labels CSV, Chroma directory. |
| `configs/base.yaml` | Weak-supervision weights, label thresholds (used in older weak_label path), **retrieval `top_k`**, **labeling `positive_top_k` / `negative_random_k`**, seeds, `no_fallback`. |
| `configs/model.yaml` | **Encoder id**, training hyperparameters, checkpoint path. |

**Important:** Training reads **`configs/model.yaml`** + **`configs/paths.yaml`**. Labeling reads **`base.yaml`** + **`paths.yaml`** + **`model.yaml`** (for embedding model name).

---

## 5. Data: from raw CSV to training rows

### 5.1 Raw inputs (`configs/paths.yaml`)

- `data/raw/resumes.csv` — needs at least `resume_id`, `resume_text` (and later processed columns).  
- `data/raw/jobs.csv` — needs at least `job_id`, `job_text`.

### 5.2 Preprocess (`scripts/run_preprocess.py` → `src/data/preprocess.py`)

- Cleans/normalizes text, extracts features such as `skills` (list, serialized in CSV), experience fields, etc.  
- Writes **`data/processed/resumes_processed.csv`** and **`data/processed/jobs_processed.csv`**.

### 5.3 Weak labels (`scripts/run_labeling.py`)

Uses **`src/labeling/weak_supervision.py`** and **`src/labeling/pair_builder.py`**.

1. Load processed jobs/resumes; parse `skills` lists.  
2. Build a **cosine similarity matrix** between all job texts and all résumé texts using **`SentenceTransformer`** with model **`embedding_model_name`** from `configs/model.yaml` (same as training encoder family: `sentence-transformers/all-MiniLM-L6-v2`).  
3. For each job–résumé pair, compute  
   `weak_score = w_embed * embedding_sim + w_skill * skill_overlap`  
   from **`configs/base.yaml`**:  
   - `weights.embedding_similarity` = **0.7**  
   - `weights.skill_overlap` = **0.3**  
4. For **each job**, take the **top `positive_top_k` = 5** pairs by `weak_score` and mark **`label = 1`**.  
5. From the **remaining** candidates for that job, draw **`negative_random_k` = 15** at random and mark **`label = 0`**.  
6. Save **`data/labels/job_resume_pairs.csv`** with columns including `job_id`, `resume_id`, `label`, scores.

So: **what we train on** is **not** human “this is a match” labels; it is **noisy, heuristic pairs** designed to teach the network “likely good vs likely bad” candidates per job.

---

## 6. The trainable model (what “model” means here)

**Class:** `SiameseMatcher` — `src/models/siamese_model.py`.

**Components:**

1. **Encoder:** `transformers.AutoModel.from_pretrained(encoder_name)`  
   - `encoder_name` = `sentence-transformers/all-MiniLM-L6-v2` (same **weights family** as indexing; training **updates** these weights via fine-tuning).  
2. **Pooling:** Mean of token hidden states, masked (function `mean_pool`).  
3. **Pair fusion:** For job vector `j` and résumé vector `r`, build  
   `[ |j − r|, j ⊙ r ]`  
   (concatenate absolute difference and element-wise product), dimension `2 * hidden_size`.  
4. **Head:** `Linear → ReLU → Dropout(0.1) → Linear → 1` scalar **logit** per pair.

**Output:** One logit per pair; passed through **sigmoid** at inference for a score in (0, 1) (`src/rag/reranker.py`).

---

## 7. Training: loss, optimizer, schedule, split, outputs

**Entrypoint:** `scripts/run_train.py` → `src/models/train.py` → `train_matcher(...)`.

### 7.1 Data merged for training

- Reads **`job_resume_pairs.csv`**.  
- Joins **`job_text`** from processed jobs and **`resume_text`** from processed résumés.  
- Drops rows with missing joins; **empty merge raises**.

### 7.2 Train / validation split (**by job**)

- **`validation_size`** = **0.2** (`model.yaml`).  
- **Split is on unique `job_id`**, not on rows: all pairs for a job stay in train **or** val.  
- Requires **at least 2 unique jobs** or training errors.

### 7.3 Batches

- **`JobResumePairDataset`** (`src/models/dataset.py`): tokenizes `job_text` and `resume_text` with **`AutoTokenizer`**, `max_length` = **64**, padding **`max_length`**.  
- **`DataLoader`** batch size = **8**, shuffle train only.

### 7.4 Loss

- **`BCEWithLogitsLoss`** with `pos_weight = positive_class_weight` = **1.0** (`src/models/losses.py`).  
- Labels are **float** 0.0 / 1.0.

### 7.5 Optimizer

- **`AdamW`**, **`weight_decay`** = **0.01**.  
- **Two parameter groups:**  
  - Encoder: **`learning_rate_encoder`** = **2e-5**  
  - Classifier head: **`learning_rate_head`** = **1e-4**

### 7.6 LR schedule

- **`get_linear_schedule_with_warmup`** from Hugging Face `transformers`.  
- **`num_warmup_steps`** = **10%** of total optimizer steps (`max(1, int(0.1 * total_steps))`).  
- **`num_training_steps`** = `len(train_loader) * epochs`.

### 7.7 Epochs and seed

- **`epochs`** = **1** (configurable in `model.yaml`).  
- **`random_seed`** = **42** for Python, NumPy, PyTorch (+ CUDA if available).

### 7.8 Hardware

- Device: **`cuda` if available, else `cpu`**.

### 7.9 Checkpoint and metrics

- Best checkpoint = **lowest validation loss** over epochs.  
- Saved to **`data/artifacts/model/best_model.pt`** (PyTorch **state_dict** only).  
- Sidecar **`best_model.metrics.json`**: `best_val_loss`, per-epoch `train_loss` / `val_loss`.

### 7.10 Hyperparameter table (copy-paste from `configs/model.yaml`)

| Key | Value |
|-----|--------|
| `embedding_model_name` | `sentence-transformers/all-MiniLM-L6-v2` |
| `batch_size` | 8 |
| `max_length` | 64 |
| `epochs` | 1 |
| `learning_rate_encoder` | 2e-5 |
| `learning_rate_head` | 1e-4 |
| `weight_decay` | 0.01 |
| `validation_size` | 0.2 |
| `random_seed` | 42 |
| `positive_class_weight` | 1.0 |
| `model_output_path` | `data/artifacts/model/best_model.pt` |

---

## 8. Embeddings and Chroma (retrieval index)

**Script:** `scripts/run_embeddings.py`.

- Loads **processed** résumés, parses `skills`.  
- Encodes all **`resume_text`** with **`SentenceTransformer`** (`model.yaml` `embedding_model_name`).  
- **`build_chroma_resume_index`** (`src/embeddings/chroma_store.py`): **recreates** collection `resumes`, adds vectors + documents + metadata.  
- Persist path: **`data/artifacts/chroma`** (`paths.yaml` `chroma_dir`).

**Online / PDF upsert:** `ingest_resume_entries` uses **upsert** so new résumés can be added without rebuilding from scratch.

---

## 9. Inference: retrieve → rerank → explain

**Core function:** `recommend_candidates` — `src/rag/pipeline.py`.

1. Load `paths.yaml`, `base.yaml`, `model.yaml` (cached).  
2. **Retrieve:** `retrieve_top_k_resumes` — embed **job** with same ST model, query Chroma for **`top_k`** (from argument or `base.yaml` **`retrieval.top_k` = 25**).  
3. **Rerank:** Load `SiameseMatcher`, load **`best_model.pt`**, score pairs, sort descending by score.  
4. **Explain:** `build_explanation` — compares **skill keywords** extracted from job vs résumé (`src/nlp/skill_extractor.py`).

**API:** `POST /recommend` — `src/api/main.py`.  
**Reranker cache:** `src/rag/reranker.py` (`lru_cache` on model load).

---

## 10. Evaluation metrics

**Implemented:** `src/eval/metrics.py` — **Precision@k**, **Recall@k**, **MRR** (reciprocal rank of first relevant).  

**Script:** `scripts/run_eval.py` (uses retrieval and/or full pipeline vs label-derived positives—see that script and `src/eval/evaluate.py` for the exact protocol on your split).

Use metrics to compare **retrieval-only** vs **retrieval + rerank** on held-out structure.

---

## 11. API, UI, scripts

| Surface | Role |
|---------|------|
| **FastAPI** `src/api/main.py` | `GET /health`, `POST /recommend`, `POST /ingest/resume-pdfs`, `POST /ingest/resume-texts` |
| **Streamlit** `src/ui/product_app.py` | Product UI: match, PDF ingest, docs, tech stack, advanced runners |
| **Scripts** `scripts/run_*.py` | Preprocess, labeling, embeddings, train, eval, RAG, API, UI, benchmarks |

---

## 12. Deployment and production notes

- **Local:** API (`run_api.py`), Streamlit (`run_ui.py`), Docker Compose uses **`Dockerfile.api`**.  
- **Hugging Face Spaces:** root **`Dockerfile`** runs Streamlit on port **7860**, `sdk: docker`; long-form platform notes and limits: see **`docs/DEPLOYMENT.md`** (Space-specific steps only—no need to duplicate here beyond “read DEPLOYMENT for HF/Docker URL behavior”).

---

## 13. Stack: PyTorch vs TensorFlow

- **PyTorch:** training loop, `SiameseMatcher`, checkpointing.  
- **Hugging Face `transformers`:** `AutoModel`, `AutoTokenizer`, LR scheduler.  
- **`sentence-transformers`:** offline embeddings for Chroma and weak-supervision similarity matrix (and same model name as encoder backbone).  
- **ChromaDB:** vector index.  
- **FastAPI / Streamlit / Docker / pytest / GitHub Actions:** product and engineering hygiene.  

**TensorFlow/Keras:** not used.

**Interview cheat line:** *“Weak labels from embedding + skill heuristics; dual-tower Siamese with pairwise fusion; BCE-with-logits; retrieve with Chroma then rerank with the fine-tuned encoder.”*

---

## 14. Other documents in `docs/`

| File | What it is | When to open it |
|------|------------|-----------------|
| **`DEPLOYMENT.md`** | GitHub vs Hugging Face Docker Space, timeouts, disk, XSRF/file upload. | When you ship or debug hosting. |
| **`NON_TECH_PROJECT_GUIDE.md`** | Long narrative for non-coders (story, analogies). | Pitches to non-technical buyers. |
| **`TECH_STACK_AND_LEARNING_GUIDE.md`** | Library-by-library interview notes + `requirements.txt` lines. | Interview prep quick scan (details duplicated here in §13 and data flow). |
| **`MODEL_CARD.md`** | **Template** for model transparency (fill in for a release). | Compliance / portfolio “model card” expectations. |
| **`EXPERIMENT_REPORT.md`** | **Template** for experiment logging. | When you run formal A/B or ablations. |

**Nothing in those files is “secret” knowledge missing from this guide**; they exist for **audience** (non-tech, ops, templates), not for splitting the technical truth.

---

## Reproduce from zero (checklist)

1. Put **`data/raw/resumes.csv`** and **`data/raw/jobs.csv`** in place.  
2. `python scripts/run_preprocess.py`  
3. `python scripts/run_labeling.py`  
4. `python scripts/run_embeddings.py`  
5. `python scripts/run_train.py`  
6. Run API or Streamlit; use `scripts/run_eval.py` for offline metrics.

If any required file is missing, the project is designed to **fail loudly** (`no_fallback: true` in `base.yaml` reflects that philosophy in config).

---

*End of single source of truth. Update this file when you change `configs/*.yaml` or training code, so portfolios and interviews stay accurate.*
