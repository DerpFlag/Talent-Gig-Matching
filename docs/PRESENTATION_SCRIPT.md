# Talent–Gig Matching — Detailed technical walkthrough (presentation script)

Use this as a spoken script or recording guide. It describes **only** what the system **is**, how each part **works**, and what every major **parameter** does. It does **not** tour documentation files or tooling stories.

---

## Opening — what problem this solves

This project is a **résumé–job matching engine**. You maintain a pool of résumés and a pool of job descriptions. The system:

1. **Cleans** raw text and derives simple structured signals (skills, experience-related fields, entity counts).  
2. **Builds training pairs** without human labelers by **ranking** candidates per job with **embedding similarity** and **keyword skill overlap**, then labeling **top-K** as positives and **random** others as negatives.  
3. **Indexes** every résumé in **ChromaDB** as a dense vector for fast **approximate nearest-neighbor search**.  
4. **Trains** a **Siamese** neural network in **PyTorch** that takes a job and a résumé and outputs a **match score**.  
5. At **query time**, embeds the job, **retrieves** the top-K vectors from Chroma, **reranks** those K with the trained network, and adds a **rule-based explanation** of which skills from a fixed vocabulary appear in both texts.

Nothing in that query path calls an LLM to generate prose; explanations are **Deterministic skill overlap** from a dictionary in code.

---

## Repository layout (what lives where)

- **`configs/`** — YAML that drives paths, labeling, retrieval width, and training hyperparameters. Changing these files changes behavior without editing Python (until you hit code-enforced constraints).  
- **`scripts/`** — Thin CLI entrypoints: each reads YAML, calls `src/`, prints status.  
- **`src/`** — Libraries: data NLP, labeling, embedding, Chroma, PyTorch models, RAG pipeline, API, PDF parsing, UI helpers.  
- **`data/raw/`** — Input CSVs (gitignored except `.gitkeep` in public repos).  
- **`data/processed/`** — Outputs of preprocess.  
- **`data/labels/`** — CSV of (job_id, resume_id, label, scores).  
- **`data/artifacts/chroma/`** — Persistent Chroma SQLite + index files.  
- **`data/artifacts/model/best_model.pt`** — Saved **state_dict** of `SiameseMatcher` at best validation loss.  
- **`data/artifacts/model/best_model.metrics.json`** — Training loss history written next to checkpoint.  
- **`data/artifacts/eval/metrics.json`** — Offline Precision@k, Recall@k, MRR aggregates.  
- **`data/artifacts/reports/benchmark.json`** — Latency report from `run_benchmark.py`.

---

## Configuration: every key in `configs/paths.yaml`

| Key | Meaning |
|-----|---------|
| `raw_resumes_path` | Input `data/raw/resumes.csv`. Must include `resume_id`, `resume_text` at minimum for preprocess. |
| `raw_jobs_path` | Input `data/raw/jobs.csv`. Must include `job_id`, `job_text`. |
| `processed_resumes_path` | Output CSV after normalize + feature extraction. |
| `processed_jobs_path` | Same for jobs. |
| `labels_path` | Weak-supervised pairs CSV consumed by training and eval. |
| `chroma_dir` | Directory passed to `chromadb.PersistentClient` for storage. |

If you change paths here, **every** script that reads this file follows the new location. Nothing falls back silently: missing files raise.

---

## Configuration: every key in `configs/base.yaml`

| Key | Value (as shipped) | Role |
|-----|---------------------|------|
| `project_name` | `talentgig_matching` | Metadata string; not read by core training loop. |
| `random_seed` | `42` | Used in `run_labeling.py` for reproducible negative sampling. |
| `no_fallback` | `true` | Declares project policy in config; pairing logic still **raises** on empty data. |
| `weights.embedding_similarity` | `0.7` | Weight in **weak_score** = `0.7 * cos_sim + 0.3 * skill_overlap`. |
| `weights.skill_overlap` | `0.3` | Must sum to `1.0` with embedding weight or `weak_supervision` raises. |
| `thresholds.positive` | `0.62` | Used only if you call **`build_weak_labels`** (threshold-based labeling). **Current** pipeline uses **`build_topk_random_pairs`** instead, which **ignores** these thresholds and uses **top-K / random-K** only. |
| `thresholds.negative` | `0.35` | Same: relevant to `build_weak_labels`, not to `run_labeling.py` as wired. |
| `retrieval.top_k` | `25` | Default width of the candidate list for **retrieval**, **`recommend_candidates`**, **`run_retrieval.py`**, and **evaluation** unless overridden by API/argument. |
| `labeling.positive_top_k` | `5` | For **each job**, the **5** highest **weak_score** pairs become **label = 1**. |
| `labeling.negative_random_k` | `15` | From **remaining** candidates after removing those 5, **15** rows sampled uniformly **without** replacement per job; **label = 0**. If fewer than 15 remain, **labeling fails explicitly**. |

**Effect of tuning `weights`:** Higher `embedding_similarity` makes cosine similarity dominate the ranking that picks positives and orders negatives; higher `skill_overlap` makes the Heuristic dictionary overlap matter more.

**Effect of `positive_top_k` / `negative_random_k`:** More positives → more (job, resume) rows per job, **heavier** label file and training set; more negatives → harder classification, slower epochs. Too few résumés per job relative to **5 + 15** causes **failure** by design.

**Effect of `retrieval.top_k`:** Larger K → more résumés passed to the **reranker** → **slower** inference and more GPU/CPU memory in the reranker batch; **better recall** possible if the true match was not in the top few ST neighbors. Smaller K → faster, risk missing good candidates that ST retrieval ranks below K.

---

## Configuration: every key in `configs/model.yaml`

| Key | Value (as shipped) | Role |
|-----|---------------------|------|
| `embedding_model_name` | `sentence-transformers/all-MiniLM-L6-v2` | **Shared identifier** for: SentenceTransformer in labeling and `run_embeddings`; encoder + tokenizer in **training** and **reranking**. |
| `batch_size` | `8` | PyTorch `DataLoader` batch size. Larger → faster steps, more VRAM/RAM; smaller → noisier gradients, slower. |
| `max_length` | `64` | **Max tokens** per **job** and per **résumé** in training and rerank inference. Text is **truncated**; increase if you lose tail of long JDs, at **cost** of memory. |
| `epochs` | `1` | Full passes over training **pair** rows. More epochs → more steps; risk overfit on noisy weak labels. |
| `learning_rate_encoder` | `2e-5` | AdamW LR for **pretrained encoder** weights. Typical BERT-style fine-tuning range. |
| `learning_rate_head` | `1e-4` | AdamW LR for **randomly initialized MLP head**. Often higher than encoder. |
| `weight_decay` | `0.01` | L2-style regularization in AdamW. |
| `validation_size` | `0.2` | Fraction of **unique job_ids** held out for validation (not random rows). |
| `random_seed` | `42` | Seeds Python, NumPy, torch (+ CUDA if present) at train start. |
| `positive_class_weight` | `1.0` | Passed to `BCEWithLogitsLoss` as `pos_weight`. Increase if positives are **underrepresented** in batches (here classes are roughly controlled by construction). |
| `model_output_path` | `data/artifacts/model/best_model.pt` | Where **best** `state_dict` is written. |

**Changing `max_length` without retraining:** Reranker uses this at inference; if you train with 64 and later change to 128 without retraining, behavior shifts (more tokens seen cold) — you should **retrain** for a fair comparison.

**Changing `embedding_model_name` without full pipeline rerun:** Breaks consistency: Chroma index and weak labels were built with **old** ST model; encoder in `.pt` expects **new** tokenizer shape. **You must** rebuild labels, Chroma index, and **retrain** if you switch backbone.

---

## Script: `scripts/run_preprocess.py`

**Inputs:** `paths.yaml` → raw CSV paths.

**Steps:**

1. Inserts project root into `sys.path` (same pattern as other scripts).  
2. Reads both raw CSVs with **`read_csv_strict`** (fails if missing columns).  
3. `preprocess_resumes`: for each row, **`normalize_text`** on `resume_text`; **`extract_skills`** → sorted list stored as **`skills`** column; **`extract_experience_years`**; **`parse_entities`** expanded into extra numeric columns.  
4. `preprocess_jobs`: same pattern on `job_text`; **`required_experience_years`** from the same experience parser on job text.  
5. Writes processed CSVs; creates parent dirs with **`ensure_parent`**.

**Downstream expectation:** `skills` ends up serialized as strings in CSV; **`run_labeling`** and **`run_embeddings`** **parse** skills back to lists with **`ast.literal_eval`**.

---

## Script: `scripts/run_labeling.py`

**Inputs:** `paths.yaml`, `base.yaml`, `model.yaml`.

**Steps:**

1. Load processed résumés and jobs; **parse** `skills` column via **`literal_eval`** (must be Python-list literal strings).  
2. **`build_similarity_matrix`**: encode **all** job texts and **all** résumé texts with **one** `SentenceTransformer`, **L2-normalized** embeddings, multiply **job_matrix @ resume_matrix.T** → cosine similarity matrix (since normalized).  
3. **`build_topk_random_pairs`**: for each job, compute **weak_score** = `0.7 * sim + 0.3 * (|job_skills ∩ resume_skills| / |job_skills|)` with division guarded by empty job skills in `skill_overlap` helper returning 0.  
4. Sort by `weak_score` descending; **head 5** → label 1; from **tail** (everything after rank 5), **sample 15** negatives → label 0.  
5. Concatenate all jobs; write **`labels_path`** CSV.

**Important:** This produces **exactly** `5 + 15 = 20` rows per job **when** enough résumés exist; the matrix is **full** Cartesian: all job–resume pairs scored in memory — **O(#jobs × #resumes)** memory and CPU.

---

## Script: `scripts/run_embeddings.py`

**Inputs:** processed résumés, `model.yaml`, `paths.yaml`.

**Steps:**

1. Parse `skills` lists again.  
2. **`encode_texts`** from `src/embeddings/embedder.py`: loads a **fresh** `SentenceTransformer` (not the LRU-cached one used at inference in `retriever.py`), encodes **all** `resume_text` with **`normalize_embeddings=True`**, progress bar on.  
3. Builds parallel lists: string **ids**, string **documents** (full text), **metadata** dict per row with **`skills`** joined by comma.  
4. **`build_chroma_resume_index`**: creates persistent client under **`chroma_dir`**, **deletes** collection **`resumes`** if present, **creates** empty collection, **`add`** all points.

**Consequence:** **`run_embeddings` wipes** the prior Chroma collection for `resumes`. Any **PDF-upserted** IDs disappear unless you re-ingest or use a DB backup. Operational implication: full rebuild vs incremental ingest must be a **deliberate** choice.

---

## Script: `scripts/run_train.py`

Calls **`train_matcher`** with every scalar from `model.yaml` and file paths from `paths.yaml`.

Output printed: best validation loss and checkpoint path.

---

## Training internals (`src/models/train.py`) — step by step

1. **`_load_training_frame`**: inner-join labels with processed jobs and résumés on ids; **empty** result **raises**.  
2. **Split**: **`train_test_split`** on **`unique job_id`**, `test_size=validation_size`, `random_state=random_seed`. Training and validation **DataFrames** must be non-empty.  
3. **Tokenizer**: `AutoTokenizer.from_pretrained(model_name)` shared by train and val datasets.  
4. **Dataset row**: tokenizes job and résumé separately to **`max_length`**, padding **`max_length`**, returns tensors + **float label**.  
5. **Model**: `SiameseMatcher(encoder_name)` on CPU or CUDA.  
6. **Loss**: `BCEWithLogitsLoss(pos_weight=tensor([positive_class_weight]))`.  
7. **Optimizer**: **AdamW** single optimizer, **two param groups** — encoder LR `lr_encoder`, classifier LR `lr_head`, `weight_decay` shared.  
8. **Scheduler**: **Linear warmup** for **10%** of total steps, then linear decay to zero over **`len(train_loader) * epochs`** steps. **Scheduler step happens **after each batch** in training loop.  
9. **Epoch loop**: train epoch → mean train loss; eval epoch **no** optimizer, **no** scheduler step → mean val loss. **Best val** saves **`state_dict`** to **`model_output_path`**.

**Metrics file:** sibling path **`best_model.metrics.json`** with `best_val_loss` and list of `{epoch, train_loss, val_loss}`.

**What is being learned:** The network sees **pairs** of short texts and weak binary targets. It learns a **decision boundary** in **pair space** (after encoder + fusion), not a generative model of résumés.

---

## Architecture: `SiameseMatcher` (`src/models/siamese_model.py`)

- **`AutoModel`** encoder → **`last_hidden_state`**.  
- **Mean pool** over tokens, masked by attention mask.  
- Two pooled vectors **job_vec**, **resume_vec**.  
- **Fusion** vector: concatenate **`abs(job_vec - resume_vec)`** and **`job_vec * resume_vec`** → size **`2 * hidden_size`** (MiniLM hidden is 384 → fusion 768).  
- **Classifier:** `Linear(768 → 384)` → **ReLU** → **Dropout(0.1)** → **`Linear(384 → 1)`** → **logit**.

**Training prediction:** logits compared to binary labels with BCE-with-logits.

**Inference (`PairReranker.score`):** same forward; **sigmoid** on logits → **scores in (0,1)** used for **sorting** (higher = more preferred).

---

## Script: `scripts/run_retrieval.py`

Smoke test only: reads **first** processed job row, calls **`retrieve_top_k_resumes`** with **`retrieval.top_k`** from `base.yaml`, prints **resume_id** and **distance** for each hit. **No reranker**, no explanation.

---

## Script: `scripts/run_rag.py`

Reads **first** job from processed jobs, calls **`recommend_candidates`** (full pipeline), prints rerank **score**, Chroma **distance**, **matched_skills** string.

---

## Script: `scripts/run_eval.py`

Calls **`evaluate_retrieval_and_rerank`** writing **`data/artifacts/eval/metrics.json`**.

**Protocol:**

- Build **relevance** map: for each **job_id**, the set of **resume_id** where **label == 1** in **`labels_path`**.  
- For each processed job that appears in that map and has **at least one** positive:  
  - Run **retrieval** top_k (**from base.yaml**).  
  - Run **same** `rerank_candidates` as production.  
  - **Precision@k** = (# of top-k that are in relevant set) / **k**.  
  - **Recall@k** = (# of top-k that are relevant) / **|relevant|**.  
  - **MRR** = 1/rank of **first** relevant in list, or 0 if none.  
- Average those three metrics across jobs separately for **retrieval-only order** vs **rerank order**.

**Interpretation caveats:** Relevant set is **weak** positives only — metrics measure consistency with **heuristic** labels, not ground-truth hiring outcomes.

---

## Script: `scripts/run_benchmark.py`

**CLI parameters:**

| Flag | Default | Role |
|------|---------|------|
| `--sample-size` | 100 | Upper bound on number of **first** processed jobs to consider. |
| `--warmup` | 5 | First N jobs call **`recommend_candidates`** but **times are discarded** (JIT / cache warm). |
| `--top-k` | 10 | Passed into **`recommend_candidates`** for each timed call. |
| `--output` | `data/artifacts/reports/benchmark.json` | JSON with avg / min / max / p50 / p90 / p95 / p99 latency in **ms**. |

**Trimmed average:** If more than 10 latencies, drops **outer 10%** each side before one more mean.

---

## Inference path (`src/rag/pipeline.py` — `recommend_candidates`)

1. **Load YAML** via **`lru_cache`** — **changing config on disk mid-process** does not reload until process restart.  
2. **`top_k`**: function argument **wins**; else **`base_cfg["retrieval"]["top_k"]`**.  
3. **Retriever** (`retrieve_top_k_resumes`):  
   - **Cached** `SentenceTransformer` per model name (`lru_cache` max 4 names).  
   - Encode query job with **`normalize_embeddings=True`** (same normalization assumption as index build).  
   - **Cached** `get_collection(chroma_dir, "resumes")` — after **upsert**, code calls **`clear_collection_cache`** so new collection handle is fetched.  
   - **`collection.query`**, **`n_results=top_k`**.  
4. **Reranker** (`get_cached_reranker`): loads **tokenizer**, builds **`SiameseMatcher`**, **`torch.load(state_dict)`** onto CPU or CUDA.  
5. **`rerank_candidates`**: batches tokenization job repeated per résumé; forward sigmoid scores; **sort descending** by score.  
6. **`build_explanation`**: **word-boundary** regex match of a **fixed skill set** in `skill_extractor.py` on lowercased job and récuré texts; returns **intersection**, **job-not-in-resume**, and **count**.

**What is *not* run at inference:** No backward pass, no gradient, **`model.eval()`** and **`torch.no_grad()`** in reranker.

---

## PDF upload and API ingest — **no automatic retraining**

**Streamlit (`product_app.py`)** file uploader:

1. Reads bytes per file.  
2. **`extract_text_from_pdf`**: `pypdf.PdfReader`, concatenate **extract_text()** per page. **Empty** or **image-only** PDF **raises** — no OCR fallback.  
3. New id: **`upload_` + UUID hex**.  
4. **`ingest_resume_entries`**: encodes texts with **`encode_texts_normalized`** (cached ST), builds metadata **skills** string from same extractor as explanations, merges optional **filename** / **source**, **`upsert_resume_documents`** into Chroma, **`clear_collection_cache`**.

**Does PDF upload retrain the Siamese model?** **No.** Only **Chroma** gains new vectors. The **`best_model.pt`** weights are **unchanged**. Reranker still scores (job, resume) using weights trained on **previous weak labels**. If new résumés use vocabulary or layout outside training distribution, **rerank quality** can drift until you **re-run labeling + train** on an updated corpus.

**FastAPI `POST /ingest/resume-pdfs`:** same ingest path; validates **`.pdf`** extension; 400 on bad PDF text.

**FastAPI `POST /ingest/resume-texts`:** JSON batch; **client-supplied** `resume_id` and text; same **upsert**.

**FastAPI `POST /recommend`:** body validated by Pydantic — **`job_description` min length 10**; **`top_k`** optional positive int.

---

## Local hosting

**API:** `python scripts/run_api.py` → **Uvicorn** serves **`src.api.main:app`** on **0.0.0.0:8000**.  
**Streamlit product UI:** `python scripts/run_ui.py` → **port 8501**, **`product_app.py`**.

**Docker Compose (`docker-compose.yml`):**  
Both **api** and **ui** services **build from `Dockerfile.api`** (same image layers), override **command** — API runs **`run_api.py`**, UI runs **`run_ui.py`**. **Port map** 8000:8000 and 8501:8501. **Volume** mounts project into `/app` for live edit; not how production usually runs.

**Root `Dockerfile` (Streamlit for Spaces):**  
Installs **`requirements-space.txt`** only (drops pytest, fastapi bundle pieces not needed for *container* UI-only path — actually space file lists chromadb torch ST etc.). Runs **`streamlit run src/ui/product_app.py`** on **7860**, **`PYTHONPATH=/home/user/app`**, thread env caps, **`--server.enableXsrfProtection false`** for iframe uploads on HF, **`fileWatcherType none`**.

---

## Hugging Face Space hosting (typical deployment for public demo)

- **Repository type:** Hugging Face **Space** with **Docker SDK**.  
- **Build:** HF builds the root **`Dockerfile`**; README YAML sets **`app_port: 7860`**, **`startup_duration_timeout`** extended for heavy imports.  
- **Runtime:** Single container, **ephemeral filesystem** on free tier — **Chroma** and **`best_model.pt`** are **lost on sleep/restart** unless you bake artifacts in (large) or use paid persistence.  
- **Network:** User hits **`*.hf.space`**; Streamlit runs behind HF proxy; **XSRF** must be **off** for **multipart** uploads in iframe cookie model.

**GitHub** stores source; **Space** clones or mirrors **`spaces/username/repo`** git — may track same commits as GitHub via manual **`push`** to HF remote.

---

## Streamlit product UI behavior (runtime UX)

- **Sidebar radio** selects **one** of: How it works, Complete walkthrough, Match, PDF ingest, Advanced pipeline. **Only that branch executes** — avoids loading **torch** on initial paint when you stay on light pages (important on small instances).  
- **Match** lazy-imports **`recommend_candidates`**.  
- **PDF ingest** lazy-imports PDF + ingest.  
- **Advanced** can subprocess **`run_preprocess.py`**, **`run_labeling.py`**, etc., with long timeouts — those are **full** offline jobs inside the same machine.

---

## Parameter change cheat sheet (what to expect)

| You change | Effect |
|------------|--------|
| `labeling.positive_top_k` / `negative_random_k` | Row count per job; training distribution; eval relevance sets. |
| `weights.*` | Who becomes “positive” under weak supervision. |
| `retrieval.top_k` | Width of candidate funnel; latency; recall vs precision tradeoff. |
| `batch_size`, `max_length`, LRs, `epochs` | Training dynamics; checkpoint quality; train time. |
| `embedding_model_name` | **Requires** relabel + re-embed + **retrain** for internal consistency. |
| PDF uploads | **Chroma only**; reranker frozen until you **manually** retrain. |

---

## Closing line (spoken)

**Talent–Gig Matching** is a **concrete, inspectable** ML system: **weak-supervised pair generation**, **sentence embedding** indexing in **Chroma**, **fine-tuned Siamese reranking** in **PyTorch**, **deterministic skill explanations**, **REST** and **Streamlit** surfaces, **offline** metrics and **latency** benchmarks — with every stage governed by **checked-in YAML** and **scripts** you can run line by line.

Thank you.
