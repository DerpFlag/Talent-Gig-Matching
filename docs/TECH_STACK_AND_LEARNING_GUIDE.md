# Tech stack, libraries, and interview map

> **Complete end-to-end walkthrough** (data → labels → training → inference): **[`COMPLETE_PROJECT_GUIDE.md`](COMPLETE_PROJECT_GUIDE.md)**.  
> Below is a **compact** library and interview supplement.

---

This project is a **local-first, production-style** pipeline for matching résumés to job descriptions. The trainable matcher and much of the NLP stack use **PyTorch** and **Hugging Face** libraries.

Your two goals:

1. **Showcase / monetizable demo** — packaged API, Docker, Streamlit UI, PDF ingest, evaluation metrics, documentation, CI, deploy story (e.g. Hugging Face Docker Space).
2. **Learning / interviews** — use the tables below to explain *what* each piece does and *why* it appears in a serious ML system.

---

## Big picture (one paragraph)

**Offline:** Clean text → build weak labels → embed résumés into **Chroma** → train a **Siamese-style** reranker in **PyTorch** (encoder from **Transformers**). **Online:** Embed the job query, **retrieve** top‑k from Chroma, **rerank** with the trained model, **explain** overlaps with a small rule-based skill layer. That “retrieve then refine then explain” pattern is what people often call **RAG-style** matching (retrieval + optional LLM-style steps; here the “brain” after retrieval is your reranker, not a generative LLM).

---

## Deep learning & Hugging Face

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **PyTorch** | Industry-standard tensor / autograd framework for deep learning | `SiameseMatcher` forward pass, loss, `DataLoader`, training loop, `torch.load` for `best_model.pt` | “Training loop, GPU/CPU, checkpoints, `no_grad()` at inference.” |
| **transformers (Hugging Face)** | Library providing pretrained **AutoModel**, **AutoTokenizer**, schedulers | Encoder backbone + tokenizer for the Siamese network; `get_linear_schedule_with_warmup` in training | “We fine-tune or head-train on top of a pretrained encoder; tokenizer handles subword boundaries.” |
| **sentence-transformers** | Higher-level API for sentence embedding models | Encode résumés and jobs into vectors for similarity / Chroma indexing and weak supervision signals | “Sentence embeddings for dense retrieval vs sparse BM25; normalize embeddings for cosine/IP distance.” |

---

## Retrieval, “RAG-style” flow, and storage

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **ChromaDB** | Embedded **vector database** (persisted on disk) | Store résumé embeddings + metadata; **query** with job embedding for top‑k | “ANN vs brute force; persistence; collection schema; upsert for new PDFs.” |
| **RAG-style pipeline** | **R**etrieval **A**ugmented **G**eneration pattern (often retrieval + LLM) | Here: **retrieve** in Chroma → **rerank** with PyTorch model → **explain** with skill overlap (no generative LLM in the core path) | “When to retrieve first to cut search space; reranking for precision; trade-offs vs end-to-end rankers.” |

---

## Classical ML & data science

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **NumPy** | Fast numeric arrays | Vector ops, metrics helpers where needed | “Array shapes, broadcasting, working under pandas/torch.” |
| **pandas** | Tabular data in Python | CSV I/O, preprocessing tables, labels, eval joins | “ETL, groupby, merging ground truth for offline metrics.” |
| **scikit-learn** | Classical ML utils | `train_test_split` for train/validation split of pairs | “Baseline models elsewhere; here lightweight splits / could extend to baselines.” |

---

## NLP & rules in this project

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **Regex + custom parsers** (`src/nlp/*`) | Pattern-based text cleanup and features | Normalize text, years of experience, entity-ish counts, **skill keyword** overlap for explanations and weak labels | “Weak supervision combines rules + embeddings; not only end-to-end black box.” |
| **Weak supervision** | No gold labels; approximate labels from heuristics | Build (job, résumé) pairs with hybrid scores, top‑k positives, random negatives | “Label noise, calibration, why you still need offline eval.” |

---

## Backend, API, and UI

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **FastAPI** | Modern Python web framework for APIs | `POST /recommend`, `POST /ingest/resume-pdfs`, etc. | “Schema validation, async uploads, OpenAPI docs.” |
| **Pydantic** | Data validation from type hints | Request/response models | “Contracts between services; fewer runtime bugs.” |
| **Uvicorn** | ASGI server | Runs the FastAPI app | “ASGI vs WSGI; production behind nginx.” |
| **Streamlit** | Quick Python-native dashboards | Product UI: match, PDF ingest, docs, pipeline buttons | “Prototyping vs React SPA; when Streamlit is enough for demos.” |
| **requests** | HTTP client | Optional calls from minimal Streamlit demo to API | “Service boundaries: UI vs API.” |
| **python-multipart** | Multipart form parsing | FastAPI file uploads | “Why it’s a separate dependency.” |
| **pypdf** | PDF text extraction | Turn résumé PDFs into plain text before embedding | “PDF is messy; OCR is a separate product decision.” |

---

## Config, quality, and MLOps-ish pieces

| Piece | What it is | How we use it | Interview angle |
|--------|------------|----------------|-----------------|
| **PyYAML** | YAML parser | `configs/*.yaml` for paths, hyperparameters, retrieval settings | “Separate config from code; twelve-factor apps.” |
| **python-dotenv** | Load `.env` into environment | Local secrets / tokens (not committed) | “Never commit secrets; env in cloud provider.” |
| **tqdm** | Progress bars | Long-running scripts | “UX for batch jobs.” |
| **pytest** | Test runner | Metrics, API validation, PDF extract behavior | “What to unit test in ML (pure functions, API contracts).” |
| **huggingface_hub** | HF Hub client | Tokens, optional Space utilities | “Model distribution, gated models.” |
| **Docker / Docker Compose** | Container images + multi-service run | `Dockerfile` (Streamlit on Spaces), `Dockerfile.api`, `docker compose` for API+UI locally | “Reproducible deploys; dev/prod parity.” |
| **GitHub Actions** | CI | Install deps, run `pytest` on push | “Shift-left testing; merge gates.” |

---

## `requirements.txt` (line-by-line purpose)

| Dependency | Role |
|------------|------|
| `numpy` | Numerics |
| `pandas` | Tables, CSV pipeline |
| `scikit-learn` | `train_test_split` |
| `pyyaml` | Config files |
| `tqdm` | Progress |
| `python-dotenv` | Environment variables |
| `sentence-transformers` | Embedding models |
| `transformers` | Tokenizer, AutoModel, schedulers |
| `torch` | Deep learning |
| `chromadb` | Vector store |
| `fastapi` | HTTP API |
| `uvicorn` | ASGI server |
| `pydantic` | Schemas |
| `streamlit` | Web UI |
| `pytest` | Tests |
| `requests` | HTTP client |
| `pypdf` | PDF text |
| `python-multipart` | Upload parsing |
| `huggingface_hub` | Hub API / auth |

*(The Space also uses `requirements-space.txt`, a slimmer subset for the Streamlit container.)*

---

## How to talk about this in interviews

- **“Why PyTorch + Hugging Face here?”**  
  Strong ecosystem for **pretrained encoders** and **fine-tuning** sentence models; fits this pipeline end-to-end.

- **“Is this real RAG?”**  
  You have **retrieval** + **downstream scoring** + **explainability**. Classic RAG adds a **generator** (LLM). You can say: “RAG-style retrieval-first design; reranker instead of LLM for deterministic, cheaper inference in this MVP.”

- **“What would you add for enterprise scale?”**  
  Auth, multi-tenant storage, job queues for training, observability (metrics/logs), canary deploys, human feedback on labels, and optional OCR for scans.

---

## Related docs

- **Master guide:** `docs/COMPLETE_PROJECT_GUIDE.md`
- Non-technical narrative: `docs/NON_TECH_PROJECT_GUIDE.md`
- Deploy story: `docs/DEPLOYMENT.md`
- Model card template: `docs/MODEL_CARD.md`
