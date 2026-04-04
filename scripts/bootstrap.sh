#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q

echo "Bootstrap complete."
echo "Next: place raw data files under data/raw and run scripts in README."
