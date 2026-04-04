"""
Create (or verify) a Hugging Face Space for Streamlit, then link GitHub in the browser.

Requires: pip install huggingface_hub
Environment: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN to a write token.

After this script succeeds, open the printed URL → Settings → Git repository
→ connect https://github.com/DerpFlag/Talent-Gig-Matching (or your fork).
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) to a Hugging Face token with write access.",
            file=sys.stderr,
        )
        sys.exit(1)
    api = HfApi(token=token)
    me = api.whoami()
    if not isinstance(me, dict) or "name" not in me:
        print("Unexpected whoami() response.", file=sys.stderr)
        sys.exit(1)
    username = str(me["name"])
    repo_id = f"{username}/talent-gig-matching"
    api.create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")
    print(
        "Next: Space Settings → Git repository → "
        "connect https://github.com/DerpFlag/Talent-Gig-Matching and pick main."
    )


if __name__ == "__main__":
    main()
