#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _build_loou_splits, _evaluate_responses, _extract_split_rows
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.ensemble import WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.mf import LowRankMFOnlineEstimator
from vocab_benchmark.estimators.neural_advanced import NeuralMemoryMIRTConfig, NeuralMemoryMIRTEstimator
from vocab_benchmark.estimators.personalized import PersonalizedKNNPriorEstimator
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

    neural_configs = [
        NeuralMemoryMIRTConfig(
            hidden_dim=128,
            ability_dim=64,
            lr=3e-4,
            dropout=0.2,
            max_epochs=12,
            seed=seed,
            weight_decay=1e-4,
            class_balance_weight=1.0,
            hard_negative_weight=0.4,
            balanced_surrogate_weight=0.7,
            early_stopping_patience=3,
            hard_negative_k=8,
            target_batch_size=512,
            min_prefix_len=16,
            max_prefix_len=192,
            prior_logit_weight=0.8,
            user_rate_weight=1.0,
            dynamic_centering_weight=1.0,
            warmup_epochs=2,
            user_rate_centering_weight=0.0,
        ),
        NeuralMemoryMIRTConfig(
            hidden_dim=128,
            ability_dim=64,
            lr=5e-4,
            dropout=0.2,
            max_epochs=16,
            seed=seed,
            weight_decay=1e-4,
            class_balance_weight=1.0,
            hard_negative_weight=0.6,
            balanced_surrogate_weight=0.8,
            early_stopping_patience=4,
            hard_negative_k=12,
            target_batch_size=512,
            min_prefix_len=16,
            max_prefix_len=256,
            prior_logit_weight=0.8,
            user_rate_weight=1.0,
            dynamic_centering_weight=1.0,
            warmup_epochs=2,
            user_rate_centering_weight=0.0,
        ),
    ]

    models = [NeuralMemoryMIRTEstimator(cfg) for cfg in neural_configs]
    models.append(
        WeightedAveragedEnsembleEstimator(
            members=[
                PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4),
                LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed),
            ],
            weights=[0.7, 0.3],
            name="ensemble_knn25_mf16_w70_30",
            logit_bias=0.0,
        )
    )

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode="neural_memory_mirt",
        splits=splits,
        estimators=models,
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=rng,
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out["query_policy"] = "fixed_global_200"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_neural_memory_mirt.csv"
    out.to_csv(out_path, index=False)
    leaderboard = (
        out.groupby(["estimator", "q"], as_index=False)[["balanced_accuracy", "accuracy", "nll", "brier", "auroc"]]
        .mean()
        .sort_values(["q", "balanced_accuracy"], ascending=[True, False])
    )
    leaderboard.to_csv(out_dir / "leaderboard_neural_memory_mirt.csv", index=False)

    best = float(leaderboard["balanced_accuracy"].max()) if not leaderboard.empty else float("nan")
    print(f"Saved: {out_path}")
    print(f"Best balanced accuracy: {best:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/neural_memory_mirt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=8)
    args = parser.parse_args()
    run(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed, max_users=args.max_users)
