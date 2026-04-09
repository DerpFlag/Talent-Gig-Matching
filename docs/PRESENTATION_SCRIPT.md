# Talent–Gig Matching — Presentation Script

Hi everyone, thank you for being here.

Today I am presenting **Talent–Gig Matching**, an end-to-end system that takes **job descriptions** and **résumés**, builds a **searchable vector index**, trains a **neural reranker**, and returns **ranked candidates** with **transparent skill explanations** — not a black-box score.

[Open Cursor project: `talentgig_project` / GitHub: **Talent-Gig-Matching**]

Before the demo, one important point up front:  
**AI-assisted development through Cursor was core to this project** — from pipeline design and refactoring to debugging imports, Docker for Hugging Face Spaces, CI, and documentation. Design and technical decisions stayed mine; the tooling accelerated iteration.

At a high level: you put **structured hiring data** (CSVs locally, or **PDF résumés** in the web app), run an ** offline pipeline** (preprocess → weak labels → embeddings → training), then at **query time** you paste a **job description** and get a **shortlist** scored by a **Siamese-style matcher** after **vector retrieval**.

---

## Opening the “why”

Recruiters and agencies drown in résumés. Keyword search misses paraphrases; reading everything does not scale. This project uses **dense embeddings** so “machine learning engineer” and “ML engineer” sit closer in vector space, then a **trained reranker** refines the list, and a small **skill-overlap explainer** tells you *why* a profile surfaced.

[Optional: open `docs/COMPLETE_PROJECT_GUIDE.md` — single technical source of truth]

---

## Demo path A — Product UI (Streamlit)

[Run locally: `python scripts/run_ui.py` **or** open public Space: **Hugging Face — `DerpFlag/talent-gig-matching`**]

This is the **product-facing Streamlit app**.

[Sidebar: show **“How it works”** first — loads light on small hosts]

Walk through the narrative for buyers: business problem, pipeline in plain language.

[Switch sidebar to **“Complete walkthrough”**]

Here the full **master guide** renders in the browser — model, data, hyperparameters, deploy map. One document so nobody chases five different READMEs.

[Switch to **“PDF ingest”**]

Upload one or more **PDF résumés**. Text is extracted with **pypdf** — **scanned-only PDFs** are out of scope unless you add OCR later; the app **fails clearly** instead of guessing.

[Switch to **“Match”** — ensure **`best_model.pt`** and **Chroma** exist or explain cold-start]

Paste a **job description**, set **top‑k**, run **matching**.  
Behind the scenes: **embed the job** → **Chroma top‑k** → **rerank with Siamese weights** → **skill explanation**.

[Optional: **“Advanced pipeline”** — show subprocess buttons: preprocess, labeling, embeddings, train — warn that train can take long on CPU]

---

## Demo path B — API (for engineers)

[Terminal 1: `python scripts/run_api.py`]

[Terminal 2: `curl` or OpenAPI docs at `/docs`]

- **`POST /recommend`** — JSON body: `job_description`, `top_k`.  
- **`POST /ingest/resume-pdfs`** — multipart PDFs for incremental index updates.

This is what you would wire into a **real ATS** or internal tool; Streamlit is the **demo shell**.

[Open `src/api/main.py` + `src/api/schemas.py`]

---

## How the pipeline works (end-to-end story)

I will walk the **happy path** the way the code does it.

**Step 1 — Raw data**  
`data/raw/resumes.csv` and `data/raw/jobs.csv` with IDs and text.

[Open `configs/paths.yaml`]

**Step 2 — Preprocess**  
`python scripts/run_preprocess.py` → `src/data/preprocess.py`  
Normalizes text, extracts **skills** and simple features → **`data/processed/*.csv`**.

**Step 3 — Weak supervision (labels without human clickers)**  
`python scripts/run_labeling.py`

- Builds a **full job×résumé similarity matrix** with **SentenceTransformer** (`all-MiniLM-L6-v2`).  
- Scores each pair: **0.7 × embedding similarity + 0.3 × skill overlap** (from `configs/base.yaml`).  
- Per job: **top 5** pairs → **positive**; **15 random** from the rest → **negative**.  
- Writes **`data/labels/job_resume_pairs.csv`**.

So the **training target** is **noisy but structured**: the model learns “plausible good vs plausible bad” for that corpus — not legally binding hiring truth.

[Open `src/labeling/pair_builder.py` briefly]

**Step 4 — Vector index**  
`python scripts/run_embeddings.py`  
Encodes every résumé, **recreates** Chroma collection **`resumes`** → **`data/artifacts/chroma`**.

[Open `src/embeddings/chroma_store.py`]

**Step 5 — Train the reranker**  
`python scripts/run_train.py` → `src/models/train.py`

- Merges labels with **job_text** and **resume_text**.  
- **Train/val split by `job_id`** — so you do not leak the same job across splits.  
- **SiameseMatcher** (`src/models/siamese_model.py`): shared **Hugging Face encoder**, **mean pooling**, fusion ** [|j−r|, j⊙r] **, small MLP head → **logit**.  
- **BCE-with-logits**, **AdamW** with **different LRs** for encoder vs head, **linear warmup schedule**.  
- Checkpoint: **`data/artifacts/model/best_model.pt`**; metrics JSON beside it.

[Open `configs/model.yaml` — batch size, max length, epochs, learning rates]

**Step 6 — Query time (RAG-style)**  
`src/rag/pipeline.py`: **retrieve** in Chroma → **rerank** with loaded weights → **`build_explanation`** from skill keywords.

Not a generative LLM pipeline — **retrieval + learned scoring + rules** — but the *shape* is what teams mean by **retrieve then refine**.

---

## Architecture and engineering choices

**Why Chroma?**  
Embedded, file-backed **vector DB** — good MVP for “similar résumés” without standing up a separate service on day one.

**Why weak supervision?**  
Real **labeled (job, hired)** pairs are rare and slow. Heuristics bootstrap a **first reranker**; you can swap in human labels later.

**Why PyTorch + Hugging Face?**  
Standard stack for **pretrained transformers** and **fine-tuning**; **`sentence-transformers`** lines up indexing, weak labels, and the encoder family.

**No-fallback policy**  
The repo is wired to **fail loudly** if files or artifacts are missing (`configs/base.yaml` **`no_fallback: true`**) — I preferred **explicit errors** over silent empty results in hiring contexts.

[Open `.cursor/rules/rigid-no-fallback.mdc` if discussing engineering culture]

---

## Deployment and CI

**GitHub** holds the source of truth. **GitHub Actions** runs **`pytest`** on push — we fixed **`import src`** with **`tests/conftest.py`** and **`PYTHONPATH`**.

[Open `.github/workflows/ci.yml`]

**Docker:**  
- **`Dockerfile.api`** — FastAPI for local compose.  
- Root **`Dockerfile`** — Streamlit on port **7860** for **Hugging Face Docker Spaces** (Streamlit is not a separate HF SDK anymore).

**Hugging Face Space**  
Public demo: attach repo, build container, **disable Streamlit XSRF** for iframe uploads, extend **startup timeout** for heavy imports. Ephemeral disk — I document limits in **`docs/DEPLOYMENT.md`**.

---

## Documentation map (for the audience)

If someone asks “where is everything?”

- **`docs/COMPLETE_PROJECT_GUIDE.md`** — **one technical master**: data → model → training → inference → eval.  
- **`docs/NON_TECH_PROJECT_GUIDE.md`** — long **story for non-coders**.  
- **`docs/PRESENTATION_SCRIPT.md`** — **this talk track**.  
- **`docs/DEPLOYMENT.md`** — cloud and Space specifics.  
- **`docs/TECH_STACK_AND_LEARNING_GUIDE.md`** — library cheat sheet for interviews.

---

## Impact and what this demonstrates

For **agencies and companies**, the impact is: a **credible MVP** — API + UI + PDF ingest + eval hooks + CI + container story — that you can **extend** toward auth, multi-tenant data, and production observability.

For **interviews**, you can defend:

- **Weak supervision** and label noise.  
- **Two-stage retrieval + rerank** vs monolithic ranking.  
- **Job-level validation split**.  
- **Trade-offs** of MiniLM-sized models vs larger encoders.

For **me as the builder**, Cursor compressed **research-to-code** cycles: path fixes for Spaces, pytest on CI, consolidating docs into **one guide**, and keeping the pipeline **honest** when data is missing.

---

## Closing

That is **Talent–Gig Matching**: from raw hiring text through **weak labels** and **Chroma** to a **trained PyTorch reranker**, exposed via **FastAPI** and **Streamlit**, with **tests in CI** and a **hosted demo** path.

Planned improvements (talk track hooks): **OCR for scans**, **stronger persistence** on free hosts, **auth**, and **periodic retraining** jobs separate from live upload.

Thank you.
