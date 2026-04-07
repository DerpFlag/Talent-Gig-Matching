"""
Company-facing web app: PDF ingestion, job matching, and embedded documentation.
Run locally: python scripts/run_ui.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import os
import subprocess

import streamlit as st

from src.ui.guide_loader import GUIDE_PATH, PROJECT_ROOT, load_guide_sections

COMPLETE_GUIDE_PATH = PROJECT_ROOT / "docs" / "COMPLETE_PROJECT_GUIDE.md"
from src.ui.pipeline_runner import LESSONS, run_script

_CSS = """
<style>
  div.block-container { padding-top: 1.2rem; max-width: 1200px; }
  .tg-hero {
    background: linear-gradient(135deg, #0f2744 0%, #1a4d7a 55%, #0d3b5c 100%);
    color: #f4f8fc;
    padding: 1.75rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 40px rgba(15, 39, 68, 0.25);
  }
  .tg-hero h1 { margin: 0 0 0.35rem 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
  .tg-hero p { margin: 0; opacity: 0.92; font-size: 1.05rem; line-height: 1.5; }
  .tg-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    background: #fafbfc;
    margin-bottom: 0.75rem;
  }
  .tg-muted { color: #64748b; font-size: 0.9rem; }
</style>
"""


def _hero() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="tg-hero"><h1>Talent–Gig Matcher</h1>'
        "<p>Upload résumés (PDF), paste a job description, and get ranked candidates "
        "with transparent skill overlap. Documentation is built in for stakeholders.</p></div>",
        unsafe_allow_html=True,
    )


def tab_match() -> None:
    from src.rag.pipeline import recommend_candidates

    st.subheader("Match candidates to a job")
    st.markdown(
        '<p class="tg-muted">Uses your Chroma index + trained reranker when present. '
        "Add PDFs in the next tab first if the pool is empty.</p>",
        unsafe_allow_html=True,
    )
    top_k = st.slider("How many candidates to retrieve", min_value=5, max_value=50, value=15)
    jd = st.text_area("Job description (JD)", height=220, placeholder="Paste the full job description…")
    if st.button("Run matching", type="primary", use_container_width=True):
        if len(jd.strip()) < 10:
            st.error("Job description is too short (minimum 10 characters).")
            return
        with st.spinner("Retrieving and reranking… first load can take a minute."):
            try:
                result = recommend_candidates(job_text=jd, top_k=int(top_k))
            except Exception as exc:
                st.error(str(exc))
                return
        cands = result.get("candidates") or []
        st.success(f"Returned {len(cands)} candidates (top_k={result.get('top_k')}).")
        for idx, row in enumerate(cands, start=1):
            exp = row.get("explanation") or {}
            matched = ", ".join(exp.get("matched_skills") or []) or "—"
            missing = ", ".join(exp.get("missing_job_skills") or []) or "—"
            with st.container():
                st.markdown(
                    f'<div class="tg-card"><b>#{idx}</b> — '
                    f"<code>{row.get('resume_id')}</code> · "
                    f"rerank <b>{float(row.get('rerank_score', 0)):.4f}</b> · "
                    f"distance <b>{float(row.get('retrieval_distance', 0)):.4f}</b></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Matched skills: {matched}")
                st.caption(f"Missing (from JD): {missing}")
                rt = str(row.get("resume_text") or "")
                st.text(rt[:900] + ("…" if len(rt) > 900 else ""))


def tab_ingest_pdf() -> None:
    from src.data.pdf_extract import extract_text_from_pdf
    from src.embeddings.ingest import ingest_resume_entries

    st.subheader("Add résumés from PDF")
    st.markdown(
        '<p class="tg-muted">Text is extracted in-process (no OCR for scanned pages). '
        "Each file is embedded and <b>upserted</b> into the same Chroma collection as batch training data. "
        "This is <b>not</b> full model retraining; it updates the searchable pool.</p>",
        unsafe_allow_html=True,
    )
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Parse & add to index", type="primary", use_container_width=True):
        pairs: list[tuple[str, str]] = []
        extras: list[dict[str, str]] = []
        errors: list[str] = []
        for f in files:
            name = f.name or "upload.pdf"
            try:
                raw = f.getvalue()
                text = extract_text_from_pdf(raw)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
                continue
            rid = f"upload_{uuid.uuid4().hex}"
            pairs.append((rid, text))
            extras.append({"filename": name[:220], "source": "pdf_ui"})
        if errors:
            for e in errors:
                st.warning(e)
        if not pairs:
            st.error("No PDFs could be ingested.")
            return
        with st.spinner("Embedding and writing to Chroma…"):
            try:
                ingest_resume_entries(pairs, extra_metadatas=extras)
            except Exception as exc:
                st.error(str(exc))
                return
        st.success(f"Indexed {len(pairs)} résumé(s).")
        st.dataframe(
            [{"resume_id": p[0], "chars": len(p[1]), "file": ex.get("filename")} for p, ex in zip(pairs, extras)],
            use_container_width=True,
        )


def tab_complete_walkthrough() -> None:
    st.subheader("Complete walkthrough — single source of truth")
    st.caption(
        "Data → weak labels → Chroma → training (PyTorch / Hugging Face) → retrieve → rerank → explain."
    )
    if not COMPLETE_GUIDE_PATH.exists():
        st.error(f"Missing `{COMPLETE_GUIDE_PATH}`.")
        return
    st.markdown(COMPLETE_GUIDE_PATH.read_text(encoding="utf-8"))


def tab_how_it_works() -> None:
    st.subheader("How this system works")
    st.markdown(
        "Sections below mirror `docs/NON_TECH_PROJECT_GUIDE.md` so hiring managers and buyers "
        "see the same story engineers use."
    )
    sections = load_guide_sections()
    titles = [s[0] for s in sections]
    idx = st.selectbox("Section", range(len(titles)), format_func=lambda i: titles[i])
    st.markdown(sections[idx][1])


def tab_advanced() -> None:
    st.subheader("Offline training pipeline")
    st.markdown(
        '<p class="tg-muted">These steps rebuild labels, retrain the reranker, and run evaluation. '
        "Typical SaaS products run them as scheduled jobs, not on every browser click.</p>",
        unsafe_allow_html=True,
    )
    for lesson in LESSONS:
        with st.expander(lesson["title"], expanded=False):
            st.markdown(lesson["teach"])
            if lesson["script"]:
                if st.button(f"Run {lesson['script']}", key=f"adv_{lesson['id']}"):
                    with st.spinner("Running…"):
                        code, log = run_script(lesson["script"], lesson["timeout"])
                    st.code(log[-12000:] if len(log) > 12000 else log, language="text")
                    if code != 0:
                        st.error(f"Exit code {code}")
                    else:
                        st.success("Done.")
            else:
                sample = st.number_input("Benchmark sample size", 1, 500, 50, key=f"bm_s_{lesson['id']}")
                warmup = st.number_input("Warm-up", 0, 50, 5, key=f"bm_w_{lesson['id']}")
                topk = st.number_input("Top-k", 1, 100, 10, key=f"bm_k_{lesson['id']}")
                if st.button("Run benchmark", key=f"bm_b_{lesson['id']}"):
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "run_benchmark.py"),
                        "--sample-size",
                        str(int(sample)),
                        "--warmup",
                        str(int(warmup)),
                        "--top-k",
                        str(int(topk)),
                    ]
                    proc = subprocess.run(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=3600,
                        env=os.environ.copy(),
                    )
                    log = (proc.stdout or "") + (proc.stderr or "")
                    st.code(log[-12000:] if len(log) > 12000 else log, language="text")


def main() -> None:
    st.set_page_config(
        page_title="Talent–Gig Matcher",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _hero()

    with st.sidebar:
        st.header("Product tour")
        st.markdown(
            "- **Match** — JD in, ranked résumés out.\n"
            "- **PDF ingest** — add candidates to the vector index.\n"
            "- **How it works** — buyer-friendly documentation.\n"
            "- **Complete walkthrough** — one doc: model, data, training, deploy map.\n"
            "- **Advanced** — full offline pipeline scripts.\n"
        )
        st.markdown("---")
        page = st.radio(
            "Current view",
            [
                "How it works",
                "Complete walkthrough",
                "Match",
                "PDF ingest",
                "Advanced pipeline",
            ],
            index=0,
            help="Only this view runs heavy ML code. Tabs would load everything at once and can crash small Spaces.",
        )
        st.markdown("---")
        st.markdown(f"**Project:** `{PROJECT_ROOT}`")
        st.markdown(f"**Guide file:** `{GUIDE_PATH}`")
        st.markdown(
            "**Deploy (free tiers):** push the repo to GitHub; optional demo on "
            "[Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit). "
            "See `docs/DEPLOYMENT.md` for limits."
        )

    if page == "How it works":
        tab_how_it_works()
    elif page == "Complete walkthrough":
        tab_complete_walkthrough()
    elif page == "Match":
        tab_match()
    elif page == "PDF ingest":
        tab_ingest_pdf()
    else:
        tab_advanced()


if __name__ == "__main__":
    main()
