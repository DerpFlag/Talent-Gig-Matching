import re


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t
