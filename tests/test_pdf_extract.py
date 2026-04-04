import pytest

from src.data.pdf_extract import extract_text_from_pdf


def test_pdf_extract_empty_bytes() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_pdf(b"")
