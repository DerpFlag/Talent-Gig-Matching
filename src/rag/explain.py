from src.nlp.skill_extractor import extract_skills


def build_explanation(job_text: str, resume_text: str) -> dict:
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    overlap = sorted(list(job_skills & resume_skills))
    missing = sorted(list(job_skills - resume_skills))
    return {
        "matched_skills": overlap,
        "missing_job_skills": missing,
        "matched_skill_count": len(overlap),
    }
