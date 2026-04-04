import re
from typing import Set


DEFAULT_SKILLS = {
    "python",
    "sql",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "docker",
    "aws",
    "fastapi",
    "flask",
    "pandas",
    "numpy",
    "spark",
}


def extract_skills(text: str, vocab: Set[str] | None = None) -> Set[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    dictionary = vocab if vocab is not None else DEFAULT_SKILLS
    found = set()
    haystack = text.lower()
    for skill in dictionary:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, haystack):
            found.add(skill)
    return found
