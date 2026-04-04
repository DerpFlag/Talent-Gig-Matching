import requests
import streamlit as st

st.set_page_config(page_title="Talent-Gig Matcher", layout="wide")
st.title("Talent-Gig Matching Demo")

api_url = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
top_k = st.number_input("Top K candidates", min_value=1, max_value=100, value=25, step=1)
job_description = st.text_area("Job Description", height=220)

if st.button("Recommend Candidates"):
    if not job_description.strip():
        st.error("Job description cannot be empty.")
    else:
        payload = {"job_description": job_description, "top_k": int(top_k)}
        response = requests.post(f"{api_url}/recommend", json=payload, timeout=120)
        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            result = response.json()
            st.success(f"Returned {len(result['candidates'])} candidates.")
            for idx, cand in enumerate(result["candidates"], start=1):
                st.markdown(
                    f"### {idx}) Resume ID: `{cand['resume_id']}` | "
                    f"Score: `{cand['rerank_score']:.4f}` | "
                    f"Distance: `{cand['retrieval_distance']:.4f}`"
                )
                matched = ", ".join(cand["explanation"]["matched_skills"])
                missing = ", ".join(cand["explanation"]["missing_job_skills"])
                st.write(f"**Matched skills:** {matched if matched else 'None'}")
                st.write(f"**Missing job skills:** {missing if missing else 'None'}")
                st.write(cand["resume_text"][:600] + ("..." if len(cand["resume_text"]) > 600 else ""))
