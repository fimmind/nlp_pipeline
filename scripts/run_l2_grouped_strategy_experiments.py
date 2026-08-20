#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import (
    _build_loou_splits,
    _build_user_discriminative_query_sequence,
    _evaluate_responses,
)
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.irt import L2AccuracyGroupedResidualIRTEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) == 0:
        raise ValueError(f"expected non-empty integer list, got: {raw}")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) == 0:
        raise ValueError(f"expected non-empty float list, got: {raw}")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) == 0:
        raise ValueError(f"expected non-empty string list, got: {raw}")
    return values


def _parse_optional_int_list(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if token == "":
            continue
        if token in {"none", "null", "-1"}:
            out.append(None)
        else:
            out.append(int(token))
    if len(out) == 0:
        raise ValueError(f"expected non-empty optional-int list, got: {raw}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/model_improvement_fasttext/l2_grouped_residual_strategies"),
    )
    parser.add_argument("--feature-set", choices=["legacy", "fasttext_only", "l2", "freq", "rich"], default="fasttext_only")
    parser.add_argument("--budgets", type=str, default="100,200,1000")
    parser.add_argument("--groups", type=str, default="8,12,16,24")
    parser.add_argument("--strategies", type=str, default="wordnet_supersense,kmeans_fasttext,hdbscan_fasttext,reduced_fasttext_simplex")
    parser.add_argument("--wordnet-modes", type=str, default="all_synsets,first_synset")
    parser.add_argument("--reduced-dims", type=str, default="12")
    parser.add_argument("--temperatures", type=str, default="0.10,0.25")
    parser.add_argument("--residual-priors", type=str, default="1.00,2.00")
    parser.add_argument("--hdbscan-min-cluster-sizes", type=str, default="40,80,120")
    parser.add_argument("--hdbscan-min-samples", type=str, default="none,10")
    parser.add_argument("--prior-var", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--n-fit-steps", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--gate-c", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()

    budgets = _parse_int_list(args.budgets)
    n_groups_grid = _parse_int_list(args.groups)
    strategy_grid = _parse_str_list(args.strategies)
    wordnet_modes = _parse_str_list(args.wordnet_modes)
    reduced_dim_grid = _parse_int_list(args.reduced_dims)
    temperature_grid = _parse_float_list(args.temperatures)
    residual_prior_grid = _parse_float_list(args.residual_priors)
    hdbscan_mcs_grid = _parse_int_list(args.hdbscan_min_cluster_sizes)
    hdbscan_ms_grid = _parse_optional_int_list(args.hdbscan_min_samples)
    sequence_len = max(max(budgets), 200)

    loaded = load_all(args.data_dir)
    responses = loaded.responses_static.copy()
    if args.max_users > 0:
        keep_users = sorted(responses["user_id"].astype(str).unique().tolist())[: args.max_users]
        responses = responses[responses["user_id"].astype(str).isin(set(keep_users))].copy()

    words = loaded.words.copy()
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency, feature_set=args.feature_set)
    word_index = build_word_index(words)
    user_ids = sorted(responses["user_id"].astype(str).unique().tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    resp_frame = build_response_frame(responses, user_index, word_index)
    query_sequence = _build_user_discriminative_query_sequence(resp_frame, sequence_len=sequence_len, seed=args.seed)
    splits = _build_loou_splits(user_ids)

    accuracy_values = pd.to_numeric(loaded.frequency.get("accuracy"), errors="coerce").to_numpy(dtype=np.float64)
    accuracy_values = accuracy_values[: len(words)]

    estimators: list[object] = []

    for strategy in strategy_grid:
        for n_groups in n_groups_grid:
            for temperature in temperature_grid:
                for residual_prior in residual_prior_grid:
                    if strategy == "wordnet_supersense":
                        for wordnet_mode in wordnet_modes:
                            estimator = L2AccuracyGroupedResidualIRTEstimator(
                                prior_var=args.prior_var,
                                lr=args.lr,
                                n_fit_steps=args.n_fit_steps,
                                n_groups=n_groups,
                                grouping_strategy=strategy,
                                group_temperature=temperature,
                                residual_prior_var=residual_prior,
                                embedding_dim=args.embedding_dim,
                                random_state=args.seed,
                                kmeans_n_init=args.kmeans_n_init,
                                gate_c=args.gate_c,
                                hdbscan_min_cluster_size=80,
                                hdbscan_min_samples=None,
                                wordnet_mode=wordnet_mode,
                                reduced_dim=n_groups,
                                accuracy_values=accuracy_values,
                                word_list=words["word"].astype(str).tolist(),
                            )
                            estimator.name = (
                                f"l2_grouped_{strategy}_{wordnet_mode}_g{n_groups}_"
                                f"t{int(round(temperature * 100)):03d}_rp{int(round(residual_prior * 100)):03d}"
                            )
                            estimators.append(estimator)
                    elif strategy == "hdbscan_fasttext":
                        for min_cluster_size in hdbscan_mcs_grid:
                            for min_samples in hdbscan_ms_grid:
                                ms_tag = "none" if min_samples is None else str(int(min_samples))
                                estimator = L2AccuracyGroupedResidualIRTEstimator(
                                    prior_var=args.prior_var,
                                    lr=args.lr,
                                    n_fit_steps=args.n_fit_steps,
                                    n_groups=n_groups,
                                    grouping_strategy=strategy,
                                    group_temperature=temperature,
                                    residual_prior_var=residual_prior,
                                    embedding_dim=args.embedding_dim,
                                    random_state=args.seed,
                                    kmeans_n_init=args.kmeans_n_init,
                                    gate_c=args.gate_c,
                                    hdbscan_min_cluster_size=min_cluster_size,
                                    hdbscan_min_samples=min_samples,
                                    wordnet_mode="all_synsets",
                                    reduced_dim=n_groups,
                                    accuracy_values=accuracy_values,
                                    word_list=words["word"].astype(str).tolist(),
                                )
                                estimator.name = (
                                    f"l2_grouped_{strategy}_g{n_groups}_"
                                    f"t{int(round(temperature * 100)):03d}_rp{int(round(residual_prior * 100)):03d}_"
                                    f"mcs{int(min_cluster_size)}_ms{ms_tag}"
                                )
                                estimators.append(estimator)
                    elif strategy == "reduced_fasttext_simplex":
                        for reduced_dim in reduced_dim_grid:
                            if reduced_dim != n_groups:
                                continue
                            estimator = L2AccuracyGroupedResidualIRTEstimator(
                                prior_var=args.prior_var,
                                lr=args.lr,
                                n_fit_steps=args.n_fit_steps,
                                n_groups=n_groups,
                                grouping_strategy=strategy,
                                group_temperature=temperature,
                                residual_prior_var=residual_prior,
                                embedding_dim=args.embedding_dim,
                                random_state=args.seed,
                                kmeans_n_init=args.kmeans_n_init,
                                gate_c=args.gate_c,
                                hdbscan_min_cluster_size=80,
                                hdbscan_min_samples=None,
                                wordnet_mode="all_synsets",
                                reduced_dim=reduced_dim,
                                accuracy_values=accuracy_values,
                                word_list=words["word"].astype(str).tolist(),
                            )
                            estimator.name = (
                                f"l2_grouped_{strategy}_g{n_groups}_d{int(reduced_dim)}_"
                                f"t{int(round(temperature * 100)):03d}_rp{int(round(residual_prior * 100)):03d}"
                            )
                            estimators.append(estimator)
                    else:
                        estimator = L2AccuracyGroupedResidualIRTEstimator(
                            prior_var=args.prior_var,
                            lr=args.lr,
                            n_fit_steps=args.n_fit_steps,
                            n_groups=n_groups,
                            grouping_strategy=strategy,
                            group_temperature=temperature,
                            residual_prior_var=residual_prior,
                            embedding_dim=args.embedding_dim,
                            random_state=args.seed,
                            kmeans_n_init=args.kmeans_n_init,
                            gate_c=args.gate_c,
                            hdbscan_min_cluster_size=80,
                            hdbscan_min_samples=None,
                            wordnet_mode="all_synsets",
                            reduced_dim=n_groups,
                            accuracy_values=accuracy_values,
                            word_list=words["word"].astype(str).tolist(),
                        )
                        estimator.name = (
                            f"l2_grouped_{strategy}_g{n_groups}_"
                            f"t{int(round(temperature * 100)):03d}_rp{int(round(residual_prior * 100)):03d}"
                        )
                        estimators.append(estimator)

    print(f"Running {len(estimators)} grouped-residual L2 estimators...")
    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode="l2_grouped_residual_strategies",
        splits=splits,
        estimators=estimators,
        policies=[UniformRandomPolicy()],
        budgets=budgets,
        rng=np.random.default_rng(args.seed),
        fixed_query_sequence=query_sequence,
    )
    out["query_policy"] = "fixed_global_sequence"

    summary = (
        out.groupby(["estimator", "q"], as_index=False)[["balanced_accuracy", "accuracy", "nll", "brier", "auroc", "runtime_seconds"]]
        .mean()
        .sort_values(["q", "balanced_accuracy"], ascending=[True, False])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "l2_grouped_residual_all.csv"
    summary_path = args.out_dir / "l2_grouped_residual_summary.csv"
    out.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    for q in sorted(set(budgets)):
        top_path = args.out_dir / f"l2_grouped_residual_top_q{q}.csv"
        summary[summary["q"] == q].head(20).to_csv(top_path, index=False)
        print(f"saved {top_path}")

    best_by_estimator = (
        summary.sort_values(["balanced_accuracy", "nll"], ascending=[False, True])
        .groupby("estimator", as_index=False)
        .head(1)
        .sort_values("balanced_accuracy", ascending=False)
    )
    best_path = args.out_dir / "l2_grouped_residual_best_models.csv"
    best_by_estimator.to_csv(best_path, index=False)
    print(f"saved {raw_path}")
    print(f"saved {summary_path}")
    print(f"saved {best_path}")


if __name__ == "__main__":
    main()
