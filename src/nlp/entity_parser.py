import re


EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b")


def parse_entities(text: str) -> dict[str, int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return {
        "email_count": len(EMAIL_RE.findall(text)),
        "phone_count": len(PHONE_RE.findall(text)),
        "url_count": len(URL_RE.findall(text)),
    }
