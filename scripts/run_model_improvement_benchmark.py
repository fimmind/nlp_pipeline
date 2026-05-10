#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _build_loou_splits, _evaluate_responses, _extract_split_rows
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.baselines import DifficultyStratifiedBetaEstimator, UserRateDifficultyEstimator
from vocab_benchmark.estimators.ensemble import WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.mf import LowRankMFOnlineEstimator
from vocab_benchmark.estimators.neural import AveragedEnsembleEstimator
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
    validation_users = set()
    if "static_validation_users" in loaded.splits:
        validation_users = set(map(str, loaded.splits["static_validation_users"].get("user_ids", [])))
    if max_users > 0 and validation_users:
        validation_users = {u for u in validation_users if u in allowed}
    validation_splits = [s for s in splits if str(s["test_user_id"]) in validation_users]
    test_splits = [s for s in splits if str(s["test_user_id"]) not in validation_users]
    if not validation_splits:
        validation_splits = splits[: max(1, len(splits) // 4)]
        validation_ids = {str(s["test_user_id"]) for s in validation_splits}
        test_splits = [s for s in splits if str(s["test_user_id"]) not in validation_ids]
    if not test_splits:
        test_splits = splits

    knn_25 = PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4)
    mf_16 = LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)

    estimators = [
        knn_25,
        UserRateDifficultyEstimator(alpha=1.0, beta=1.0, blend=0.6),
        DifficultyStratifiedBetaEstimator(alpha=1.0, beta=1.0, n_bins=5),
        mf_16,
        AveragedEnsembleEstimator(members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)], name="ensemble_knn25_mf16", temperature=1.0),
        WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)],
            weights=[0.7, 0.3],
            name="ensemble_knn25_mf16_w70_30",
        ),
        WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)],
            weights=[0.7, 0.3],
            name="ensemble_knn25_mf16_w70_30_bias_p20",
            logit_bias=-0.2,
        ),
        WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)],
            weights=[0.7, 0.3],
            name="ensemble_knn25_mf16_w70_30_bias_m20",
            logit_bias=0.2,
        ),
        WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)],
            weights=[0.8, 0.2],
            name="ensemble_knn25_mf16_w80_20",
        ),
        WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(n_neighbors=25, alpha=1.0, beta=1.0, prior_blend=0.4), LowRankMFOnlineEstimator(rank=16, n_epochs=3, lr=0.03, reg=0.01, seed=seed)],
            weights=[0.6, 0.4],
            name="ensemble_knn25_mf16_w60_40",
        ),
    ]

    def tuned_estimators() -> list[object]:
        tuned = []
        for est in estimators:
            if hasattr(est, "logit_bias") and hasattr(est, "members") and validation_splits:
                best_bias = 0.0
                best_ba = -1.0
                for bias in np.linspace(-0.4, 0.4, 9):
                    est.logit_bias = float(bias)
                    val = _evaluate_responses(
                        responses=responses,
                        words=words,
                        x_words=x_words,
                        embedding_backend=loaded.embedding_backend,
                        dataset_name="responses_static",
                        data_mode="static_fixed200_improvement_validation",
                        splits=validation_splits,
                        estimators=[est],
                        policies=[UniformRandomPolicy()],
                        budgets=[100],
                        rng=rng,
                        fixed_query_sequence=fixed_query_sequence,
                        max_candidate_words_per_user=None,
                    )
                    ba = float(val["balanced_accuracy"].mean()) if not val.empty else -1.0
                    if ba > best_ba:
                        best_ba = ba
                        best_bias = float(bias)
                est.logit_bias = best_bias
            tuned.append(est)
        return tuned

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode="static_fixed200_improvement",
        splits=test_splits,
        estimators=tuned_estimators(),
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=rng,
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out["query_policy"] = "fixed_global_200"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_improvement.csv"
    out.to_csv(out_path, index=False)

    leaderboard = (
        out.groupby(["estimator", "q"], as_index=False)[["balanced_accuracy", "accuracy", "nll", "brier", "auroc"]]
        .mean()
        .sort_values(["q", "balanced_accuracy"], ascending=[True, False])
    )
    leaderboard.to_csv(out_dir / "leaderboard_improvement.csv", index=False)

    # queries_to_target for BA thresholds
    rows = []
    for est, g in leaderboard.groupby("estimator"):
        gg = g.sort_values("q")
        for t in [0.90, 0.95]:
            hit = gg[gg["balanced_accuracy"] >= t]
            q_val = float(hit["q"].min()) if not hit.empty else np.nan
            rows.append({"estimator": est, "target_balanced_accuracy": t, "queries_to_target": q_val})
    pd.DataFrame(rows).to_csv(out_dir / "queries_to_target_improvement.csv", index=False)

    best = float(leaderboard["balanced_accuracy"].max()) if not leaderboard.empty else float("nan")
    print(f"Saved: {out_path}")
    print(f"Best balanced accuracy: {best:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/improvement"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()
    run(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed, max_users=args.max_users)


if __name__ == "__main__":
    main()
