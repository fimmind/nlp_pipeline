from __future__ import annotations

from pathlib import Path


def run_benchmark(data_dir: Path, reports_dir: Path, seed: int, max_users: int):
    from .benchmark import run_benchmark as _run_benchmark

    return _run_benchmark(data_dir=data_dir, reports_dir=reports_dir, seed=seed, max_users=max_users)


__all__ = ["run_benchmark"]
