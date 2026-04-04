import pandas as pd

from src.nlp.entity_parser import parse_entities
from src.nlp.experience_parser import extract_experience_years
from src.nlp.normalizer import normalize_text
from src.nlp.skill_extractor import extract_skills


def preprocess_resumes(df: pd.DataFrame) -> pd.DataFrame:
    required = {"resume_id", "resume_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing resume columns: {missing}")
    out = df.copy()
    out["resume_text"] = out["resume_text"].apply(normalize_text)
    out["skills"] = out["resume_text"].apply(lambda x: sorted(list(extract_skills(x))))
    out["experience_years"] = out["resume_text"].apply(extract_experience_years)
    entities = out["resume_text"].apply(parse_entities).apply(pd.Series)
    out = pd.concat([out, entities], axis=1)
    return out


def preprocess_jobs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "job_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing job columns: {missing}")
    out = df.copy()
    out["job_text"] = out["job_text"].apply(normalize_text)
    out["skills"] = out["job_text"].apply(lambda x: sorted(list(extract_skills(x))))
    out["required_experience_years"] = out["job_text"].apply(extract_experience_years)
    entities = out["job_text"].apply(parse_entities).apply(pd.Series)
    out = pd.concat([out, entities], axis=1)
    return out
