#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vocab_benchmark.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()
    paths = run_benchmark(data_dir=args.data_dir, reports_dir=args.reports_dir, seed=args.seed, max_users=args.max_users)
    print(f"Saved results: {paths}")


if __name__ == "__main__":
    main()
