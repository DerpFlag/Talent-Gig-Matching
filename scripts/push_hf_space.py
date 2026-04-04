"""
One-shot: push current branch to Hugging Face Space (Docker) over HTTPS.

Requires environment variable HF_TOKEN with write access (see HF token settings).
Does not write the token into .git/config.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SPACE_GIT = "https://huggingface.co/spaces/DerpFlag/talent-gig-matching.git"
HF_USER = "DerpFlag"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) to a write-capable Hugging Face token.", file=sys.stderr)
        sys.exit(1)
    root = Path(__file__).resolve().parents[1]
    clean = SPACE_GIT.removeprefix("https://")
    push_url = f"https://{HF_USER}:{token}@{clean}"
    normal = subprocess.run(["git", "-C", str(root), "push", push_url, "main:main"], text=True)
    if normal.returncode == 0:
        raise SystemExit(0)
    forced = subprocess.run(
        ["git", "-C", str(root), "push", push_url, "main:main", "--force"],
        text=True,
    )
    raise SystemExit(forced.returncode)


if __name__ == "__main__":
    main()
