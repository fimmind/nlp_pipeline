#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _evaluate_responses
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.calibrated import OnlineThresholdWordPriorEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def _build_estimator(name: str) -> OnlineThresholdWordPriorEstimator:
    specs = {
        "min20_blend50_t1": (20, 0.5, 1.0),
        "min20_blend75_t1": (20, 0.75, 1.0),
        "min20_blend100_t1": (20, 1.0, 1.0),
        "min50_blend50_t1": (50, 0.5, 1.0),
        "min50_blend75_t1": (50, 0.75, 1.0),
        "min50_blend100_t1": (50, 1.0, 1.0),
        "min50_blend75_t08": (50, 0.75, 0.8),
        "min100_blend75_t1": (100, 0.75, 1.0),
        "min100_blend100_t1": (100, 1.0, 1.0),
    }
    if name not in specs:
        raise ValueError(f"unknown calibrated prior config: {name}")
    min_observations, threshold_blend, temperature = specs[name]
    estimator = OnlineThresholdWordPriorEstimator(
        min_observations=min_observations,
        threshold_blend=threshold_blend,
        temperature=temperature,
    )
    estimator.name = f"online_threshold_prior_{name}"
    return estimator


def _leaderboard(out: pd.DataFrame) -> pd.DataFrame:
    metrics = ["balanced_accuracy", "accuracy", "nll", "brier", "auroc"]
    return out.groupby(["estimator", "q"], as_index=False)[metrics].mean().sort_values(
        ["q", "balanced_accuracy", "nll"],
        ascending=[True, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()

    seed = 42
    loaded = load_all(Path("data"))
    responses = loaded.responses_static.copy()
    users = sorted(responses["user_id"].astype(str).unique().tolist())[: args.max_users]
    responses = responses[responses["user_id"].astype(str).isin(set(users))].copy()

    words = loaded.words
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency)
    word_index = build_word_index(words)
    user_index = {u: i for i, u in enumerate(sorted(responses["user_id"].astype(str).unique().tolist()))}
    resp_frame = build_response_frame(responses, user_index, word_index)
    fixed_query_sequence = _build_fixed_query_sequence(resp_frame, x_words, 200, seed)
    splits = [{"split_id": f"loou_{u}", "test_user_id": u} for u in users]
    estimators = [_build_estimator(name) for name in args.configs]

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode="calibrated_prior_fixed200",
        splits=splits,
        estimators=estimators,
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=np.random.default_rng(seed),
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "results_calibrated_prior.csv", index=False)
    _leaderboard(out).to_csv(args.out_dir / "leaderboard_calibrated_prior.csv", index=False)


if __name__ == "__main__":
    main()
