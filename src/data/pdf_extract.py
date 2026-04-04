from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(data: bytes) -> str:
    if not data:
        raise ValueError("PDF bytes are empty")
    reader = PdfReader(BytesIO(data))
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages")
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(
            "No extractable text in PDF. Scanned/image-only PDFs need OCR; this build does not run OCR."
        )
    return text
