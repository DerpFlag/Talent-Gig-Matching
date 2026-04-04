import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import recommend_candidates
from src.utils.io import read_csv_strict


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    return float(np.percentile(np.array(values, dtype=float), q, method="linear"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval+rereank latency benchmark.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of rows to sample from processed jobs.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up queries excluded from metrics.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k candidates for recommend_candidates.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/artifacts/reports/benchmark.json",
        help="Output JSON path for benchmark report.",
    )
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    return args


def main() -> None:
    args = parse_args()
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    jobs = read_csv_strict(paths["processed_jobs_path"])
    if "job_text" not in jobs.columns:
        raise ValueError("processed_jobs_path must contain job_text")
    sample_size = min(int(args.sample_size), len(jobs))
    warmup_count = min(int(args.warmup), sample_size)
    sample = jobs.head(sample_size).copy()
    if sample.empty:
        raise ValueError("No rows available for benchmarking.")

    # Warm-up calls are intentionally excluded from measurement.
    for _, row in sample.head(warmup_count).iterrows():
        recommend_candidates(job_text=str(row["job_text"]), top_k=int(args.top_k))

    latencies = []
    for _, row in sample.iloc[warmup_count:].iterrows():
        t0 = time.perf_counter()
        recommend_candidates(job_text=str(row["job_text"]), top_k=int(args.top_k))
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
    if not latencies:
        raise ValueError("No measured benchmark queries after warm-up.")

    sorted_latencies = sorted(latencies)
    trim_n = int(0.1 * len(sorted_latencies))
    if len(sorted_latencies) > 10 and trim_n > 0:
        trimmed = sorted_latencies[trim_n:-trim_n]
    else:
        trimmed = sorted_latencies
    report = {
        "top_k": int(args.top_k),
        "sample_size_total": sample_size,
        "warmup_queries": warmup_count,
        "queries": len(latencies),
        "latency_ms_avg": sum(latencies) / len(latencies),
        "latency_ms_avg_trimmed_10pct": float(sum(trimmed) / len(trimmed)),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p90": percentile(latencies, 90),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_p99": percentile(latencies, 99),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
