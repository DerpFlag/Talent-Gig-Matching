# Deployment: GitHub, free hosting, and what “real” SaaS adds

## GitHub

**GitHub stores code.** It does not run your FastAPI + Streamlit stack by itself. Typical flow:

1. Push this repository to GitHub.
2. Run the app locally or on a host (below) and point buyers at that URL.

Use private repos if résumés or configs must not be public.

## Hugging Face Spaces (free Streamlit demo)

[Spaces](https://huggingface.co/docs/hub/spaces) can run a **Streamlit** app on a **free CPU** machine with **ephemeral disk**.

**Fit for:** demos, portfolios, internal prototypes.

**Poor fit for:** heavy training on every upload, large persistent corpora, strict SLAs, or regulated data—free spaces sleep, have memory/time limits, and storage resets.

### Steps (recommended: GitHub as source of truth)

1. Push this repository to GitHub (e.g. [DerpFlag/Talent-Gig-Matching](https://github.com/DerpFlag/Talent-Gig-Matching)).
2. On Hugging Face: **Create new Space** → SDK **Streamlit** → under **Duplicate / import**, choose **Import from GitHub** and select the repo and branch `main`.
3. The Space reads **`README.md` YAML** at the top of this repo (`app_file: src/ui/product_app.py`). No extra “App file” typing is needed if import worked.
4. In the Space **Settings → Repository secrets**, add `HF_TOKEN` only if you must download gated models (default MiniLM is public).
5. **Cold start:** first model download can take several minutes on free CPU.
6. **Chroma + `best_model.pt`:** a new Space has an empty `data/artifacts/`. Use **PDF ingest** in the app to build vectors for uploaded résumés. **Match** still needs a trained reranker at `data/artifacts/model/best_model.pt` — train locally, then attach persistent storage or document that buyers run training in their own environment (free Space resets disk when it sleeps).

**CLI helper (optional):** with `HF_TOKEN` set, run `pip install huggingface_hub` then `python scripts/hf_create_space.py` to create an empty Space under your HF user, then connect the GitHub repo in Space settings.

### Why Streamlit on Spaces (vs API + separate frontend here)

One **Streamlit** app on a free Space is the standard low‑friction demo: a single build, no CORS, no second host. The **FastAPI** service remains in the repo for later (Docker, Railway, internal tools); running API + SPA as two free tiers is more moving parts for the same portfolio goal.

Example README header for a Space (adjust names as needed):

```yaml
---
title: Talent-Gig Matcher
emoji: briefcase
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: src/ui/product_app.py
pinned: false
---
```

## Product behavior vs “full SaaS”

| Capability | This repo (MVP) | Typical paid B2B SaaS |
|------------|-----------------|------------------------|
| PDF text extract | Yes (`pypdf`; no OCR) | OCR + human review options |
| Add résumés to index | Yes (Chroma **upsert**) | Multi-tenant DB + object storage |
| Retrain reranker on demand | Offline scripts / Advanced tab | Job queue, GPUs, versioning |
| Auth / orgs / billing | No | IdP, Stripe, admin consoles |
| Compliance (GDPR, SOC2) | Your responsibility | Vendor process + DPA |

**Important:** In this project, **uploading PDFs updates the vector index** (embed + upsert). **Training** the Siamese reranker is still a **separate, heavy** step (`run_train.py`). That matches how serious teams operate: index updates are frequent; **retraining** is scheduled.

## Other free/low-cost hosts (API)

- **Render / Railway / Fly.io** sometimes offer small free allowances; you run `uvicorn` with `Dockerfile` or a Python service.
- Expect cold starts and CPU limits similar to hobby tiers.

## What to prepare on your side

- **HF or cloud token** if you add gated model downloads.
- **Secrets** never committed: use platform “Secrets” UI for `HF_TOKEN`, etc.
- A **decision** on whether demo data is synthetic only (safest for public demos).
