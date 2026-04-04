import re


YEARS_EXPERIENCE_RE = re.compile(r"\b(\d{1,2})\+?\s*(?:years|yrs)\b")


def extract_experience_years(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matches = YEARS_EXPERIENCE_RE.findall(text.lower())
    if not matches:
        return 0
    values = [int(m) for m in matches]
    return max(values)
