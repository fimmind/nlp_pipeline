#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _build_loou_splits, _evaluate_responses, _extract_split_rows
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.neural import NeuralEncoderDecoderEstimator, NeuralEstimatorConfig
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def run(data_dir: Path, out_dir: Path, seed: int, max_users: int) -> None:
    rng = np.random.default_rng(seed)
    loaded = load_all(data_dir)
    words = loaded.words
    frequency = loaded.frequency
    embeddings = loaded.embeddings
    responses = loaded.responses_static.copy()

    users = sorted(responses["user_id"].astype(str).unique().tolist())
    if max_users > 0:
        keep = set(users[:max_users])
        responses = responses[responses["user_id"].astype(str).isin(keep)].copy()

    x_words = build_word_feature_matrix(words, embeddings, frequency)
    word_index = build_word_index(words)
    user_index = {u: i for i, u in enumerate(sorted(responses["user_id"].astype(str).unique().tolist()))}
    resp_frame = build_response_frame(responses, user_index, word_index)
    fixed_query_sequence = _build_fixed_query_sequence(resp_frame, x_words, sequence_len=200, seed=seed)

    if "static_leave_one_user_out" in loaded.splits:
        splits = _extract_split_rows(loaded.splits["static_leave_one_user_out"])
    else:
        splits = _build_loou_splits(sorted(responses["user_id"].astype(str).unique().tolist()))
    if max_users > 0:
        allowed = set(responses["user_id"].astype(str).unique().tolist())
        splits = [s for s in splits if str(s["test_user_id"]) in allowed]

    cfgs = [
        NeuralEstimatorConfig("gru_mlp", "teacher_forced", 64, 1e-3, 0.2, 10, seed, 1e-4, 0.02, 3),
        NeuralEstimatorConfig("lstm_bilinear", "curriculum_prefix", 128, 3e-4, 0.2, 14, seed, 1e-4, 0.02, 4),
        NeuralEstimatorConfig("residual_gru_gated", "contrastive_hard_negative", 128, 3e-4, 0.2, 14, seed, 1e-4, 0.02, 4),
    ]
    estimators = [NeuralEncoderDecoderEstimator(c) for c in cfgs]

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode="targeted_neural_improvement",
        splits=splits,
        estimators=estimators,
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=rng,
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out["query_policy"] = "fixed_global_200"

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "results_targeted_neural.csv", index=False)
    leaderboard = (
        out.groupby(["estimator", "q"], as_index=False)[["balanced_accuracy", "accuracy", "nll", "brier", "auroc"]]
        .mean()
        .sort_values(["q", "balanced_accuracy"], ascending=[True, False])
    )
    leaderboard.to_csv(out_dir / "leaderboard_targeted_neural.csv", index=False)

    print(f"Saved: {out_dir / 'results_targeted_neural.csv'}")
    print(f"Best balanced accuracy: {float(leaderboard['balanced_accuracy'].max()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/targeted_neural"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=8)
    args = parser.parse_args()
    run(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed, max_users=args.max_users)


if __name__ == "__main__":
    main()
