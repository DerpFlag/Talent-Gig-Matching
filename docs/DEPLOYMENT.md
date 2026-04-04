# Deployment: GitHub, free hosting, and what “real” SaaS adds

## GitHub

**GitHub stores code.** It does not run your FastAPI + Streamlit stack by itself. Typical flow:

1. Push this repository to GitHub.
2. Run the app locally or on a host (below) and point buyers at that URL.

Use private repos if résumés or configs must not be public.

## Hugging Face Spaces (free demo via Docker)

[Spaces](https://huggingface.co/docs/hub/spaces) can run a **Docker** app on a **free CPU** machine with **ephemeral disk**. **Streamlit is no longer a first-class Space SDK** ([configuration](https://huggingface.co/docs/hub/main/spaces-config-reference) only lists `gradio`, `docker`, and `static`). This repository runs **Streamlit inside Docker** on port **7860**, matching HF’s default `app_port`.

**Fit for:** demos, portfolios, internal prototypes.

**Poor fit for:** heavy training on every upload, large persistent corpora, strict SLAs, or regulated data—free spaces sleep, have memory/time limits, and storage resets.

### Steps (recommended: GitHub as source of truth)

1. Push this repository to GitHub (e.g. [DerpFlag/Talent-Gig-Matching](https://github.com/DerpFlag/Talent-Gig-Matching)).
2. On Hugging Face: **Create new Space** → SDK **Docker** (you will **not** see “Streamlit” as its own card—that is expected).
3. Import **from GitHub** and select this repo and branch `main`.
4. The Space reads **`README.md` YAML** at the top (`sdk: docker`, `app_port: 7860`) and builds the root **`Dockerfile`**, which starts `streamlit run src/ui/product_app.py`.
5. In the Space **Settings → Repository secrets**, add `HF_TOKEN` only if you must download gated models (default MiniLM is public).
6. **Cold start:** first model download can take several minutes on free CPU.
7. **Chroma + `best_model.pt`:** a new Space has an empty `data/artifacts/`. Use **PDF ingest** in the app to build vectors for uploaded résumés. **Match** still needs a trained reranker at `data/artifacts/model/best_model.pt` — train locally, then attach persistent storage or document that buyers run training in their own environment (free Space resets disk when it sleeps).

**CLI helper (optional):** with `HF_TOKEN` set, run `pip install huggingface_hub` then `python scripts/hf_create_space.py` to create an empty Space under your HF user, then connect the GitHub repo in Space settings.

### Why Docker on Spaces (vs API + separate frontend here)

One **container** on a free Space is the standard approach after HF’s Streamlit SDK removal: the **Dockerfile** runs Streamlit on **7860**. The **FastAPI** image is still available as **`Dockerfile.api`** (see `docker-compose.yml`) for local or paid hosts.

Example README header for a Docker Space (this repo matches this pattern):

```yaml
---
title: Talent-Gig Matcher
emoji: briefcase
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
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
