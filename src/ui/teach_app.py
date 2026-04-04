"""
Interactive teaching app: learn by doing + read the guide in the browser.
Run: python scripts/run_teacher_ui.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import streamlit as st

from src.ui.guide_loader import GUIDE_PATH, PROJECT_ROOT, load_guide_sections
from src.ui.pipeline_runner import LESSONS, run_script


def page_learn_and_run() -> None:
    st.header("Learn by doing — pipeline steps")
    st.caption("Each step below has a short lesson and a button to run that part of the project.")

    for lesson in LESSONS:
        with st.expander(f"{lesson['title']}", expanded=(lesson["id"] == 1)):
            st.markdown(lesson["teach"])
            if lesson["script"]:
                if st.button(f"Run: {lesson['script']}", key=f"run_{lesson['id']}"):
                    with st.spinner("Running… this may take a while."):
                        code, log = run_script(lesson["script"], lesson["timeout"])
                    if code == 0:
                        st.success("Finished OK.")
                    else:
                        st.error(f"Exit code {code}")
                    st.code(log[-12000:] if len(log) > 12000 else log, language="text")
            else:
                sample = st.number_input("Benchmark sample size", 1, 500, 50, key="bm_sample")
                warmup = st.number_input("Warm-up queries", 0, 50, 5, key="bm_warm")
                topk = st.number_input("Top-k", 1, 100, 10, key="bm_k")
                if st.button("Run benchmark", key="bm_go"):
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
                    with st.spinner("Benchmarking…"):
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


def page_guide_reader() -> None:
    st.header("Read the full guide (by section)")
    sections = load_guide_sections()
    titles = [s[0] for s in sections]
    choice = st.selectbox("Choose a section", range(len(titles)), format_func=lambda i: titles[i])
    st.markdown(sections[choice][1])


def page_try_api() -> None:
    st.header("Try recommendations (needs API running)")
    st.info(
        "Start the API in another terminal: `python scripts/run_api.py` "
        "then use this tab."
    )
    import requests

    api_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    top_k = st.number_input("Top K", 1, 100, 10)
    job_description = st.text_area("Job description", height=200)
    if st.button("Get recommendations"):
        if not job_description.strip():
            st.error("Enter a job description.")
        else:
            try:
                r = requests.post(
                    f"{api_url.rstrip('/')}/recommend",
                    json={"job_description": job_description, "top_k": int(top_k)},
                    timeout=300,
                )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return
            if r.status_code != 200:
                st.error(f"HTTP {r.status_code}: {r.text}")
                return
            data = r.json()
            st.success(f"{len(data.get('candidates', []))} candidates")
            for idx, c in enumerate(data.get("candidates", []), start=1):
                st.subheader(f"{idx}. `{c.get('resume_id')}` — score {c.get('rerank_score', 0):.4f}")
                exp = c.get("explanation") or {}
                st.write("Matched:", ", ".join(exp.get("matched_skills") or []) or "—")
                st.write("Missing:", ", ".join(exp.get("missing_job_skills") or []) or "—")
                txt = c.get("resume_text") or ""
                st.write(txt[:500] + ("…" if len(txt) > 500 else ""))


def main() -> None:
    st.set_page_config(page_title="Talent-Gig — Learn & Run", layout="wide")
    st.title("Talent–Gig Matching — Interactive teacher")
    st.markdown(
        "Use the sidebar to switch modes. **Learn & run** teaches each step and runs real scripts when you click."
    )

    mode = st.sidebar.radio(
        "Mode",
        [
            "Learn & run pipeline",
            "Read full guide (sections)",
            "Try API (recommendations)",
        ],
    )

    if mode == "Learn & run pipeline":
        page_learn_and_run()
    elif mode == "Read full guide (sections)":
        page_guide_reader()
    else:
        page_try_api()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Project root: `{PROJECT_ROOT}`")
    st.sidebar.caption(f"Guide: `{GUIDE_PATH}`")


if __name__ == "__main__":
    main()
