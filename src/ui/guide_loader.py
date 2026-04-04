from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = PROJECT_ROOT / "docs" / "NON_TECH_PROJECT_GUIDE.md"


def load_guide_sections() -> list[tuple[str, str]]:
    if not GUIDE_PATH.exists():
        return [("Guide missing", f"Expected file at `{GUIDE_PATH}`.")]
    text = GUIDE_PATH.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    title = "Start"
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((title, "\n".join(buf).strip()))
            title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf).strip()))
    return sections
