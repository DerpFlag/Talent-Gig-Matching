from pathlib import Path
import pandas as pd


def assert_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return p


def read_csv_strict(path: str) -> pd.DataFrame:
    p = assert_file(path)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
