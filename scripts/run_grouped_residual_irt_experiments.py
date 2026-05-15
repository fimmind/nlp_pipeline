#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import (
    _build_fixed_query_sequence,
    _build_loou_splits,
    _build_prior_uncertain_query_sequence,
    _build_user_discriminative_query_sequence,
    _evaluate_responses,
)
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.ensemble import WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.irt import GroupedResidualIRTOnlineEstimator, RaschIRTOnlineEstimator, TwoPLIRTOnlineEstimator
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


def _build_query_sequence(
    query_sequence: str, resp_frame: pd.DataFrame, x_words: np.ndarray, sequence_len: int, seed: int
) -> np.ndarray:
    if query_sequence == "difficulty":
        return _build_fixed_query_sequence(resp_frame, x_words, sequence_len=sequence_len, seed=seed)
    if query_sequence == "user_discriminative":
        return _build_user_discriminative_query_sequence(resp_frame, sequence_len=sequence_len, seed=seed)
    if query_sequence == "prior_uncertain":
        return _build_prior_uncertain_query_sequence(resp_frame, sequence_len=sequence_len, seed=seed)
    raise ValueError(f"unsupported query_sequence={query_sequence}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/model_improvement_fasttext/grouped_residual_irt")
    )
    parser.add_argument("--feature-set", choices=["legacy", "fasttext_only", "l2", "freq", "rich"], default="rich")
    parser.add_argument(
        "--query-sequence", choices=["difficulty", "user_discriminative", "prior_uncertain"], default="user_discriminative"
    )
    parser.add_argument("--budgets", type=str, default="100,200,1000")
    parser.add_argument("--groups", type=str, default="8,12,16,24,32")
    parser.add_argument("--grouping-strategies", type=str, default="kmeans_cosine,kmeans_euclidean,pca_quantile,anchor_cosine")
    parser.add_argument("--temperatures", type=str, default="0.10,0.20,0.35")
    parser.add_argument("--residual-priors", type=str, default="0.50,1.00,2.00")
    parser.add_argument("--prior-var", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--n-fit-steps", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--add-rasch-baselines", action="store_true")
    parser.add_argument("--add-twopl-ensembles", action="store_true")
    parser.add_argument("--twopl-weights", type=str, default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--twopl-ensemble-logit-biases", type=str, default="0.00")
    args = parser.parse_args()

    budgets = _parse_int_list(args.budgets)
    n_groups_grid = _parse_int_list(args.groups)
    strategy_grid = _parse_str_list(args.grouping_strategies)
    temperature_grid = _parse_float_list(args.temperatures)
    residual_prior_grid = _parse_float_list(args.residual_priors)
    twopl_weight_grid = _parse_float_list(args.twopl_weights)
    twopl_ensemble_bias_grid = _parse_float_list(args.twopl_ensemble_logit_biases)
    for weight in twopl_weight_grid:
        if weight <= 0.0 or weight >= 1.0:
            raise ValueError(f"twopl ensemble weights must be in (0,1), got {weight}")
    sequence_len = max(max(budgets), 200)

    loaded = load_all(args.data_dir)
    responses = loaded.responses_static.copy()
    if args.max_users > 0:
        keep_users = sorted(responses["user_id"].astype(str).unique().tolist())[: args.max_users]
        responses = responses[responses["user_id"].astype(str).isin(set(keep_users))].copy()

    words = loaded.words
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency, feature_set=args.feature_set)
    word_index = build_word_index(words)
    user_ids = sorted(responses["user_id"].astype(str).unique().tolist())
    user_index = {u: i for i, u in enumerate(user_ids)}
    resp_frame = build_response_frame(responses, user_index, word_index)
    query_sequence = _build_query_sequence(args.query_sequence, resp_frame, x_words, sequence_len, args.seed)
    splits = _build_loou_splits(user_ids)

    def build_grouped_estimator(strategy: str, n_groups: int, temperature: float, residual_prior: float) -> GroupedResidualIRTOnlineEstimator:
        estimator = GroupedResidualIRTOnlineEstimator(
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
            pca_components=args.pca_components,
        )
        estimator.name = (
            f"grouped_residual_irt_{strategy}_g{n_groups}_"
            f"t{int(round(temperature * 100)):03d}_rp{int(round(residual_prior * 100)):03d}"
        )
        return estimator

    estimators: list[object] = []
    grouped_specs: list[tuple[str, int, float, float, str]] = []
    for strategy in strategy_grid:
        for n_groups in n_groups_grid:
            for temperature in temperature_grid:
                for residual_prior in residual_prior_grid:
                    estimator = build_grouped_estimator(strategy, n_groups, temperature, residual_prior)
                    grouped_specs.append((strategy, n_groups, temperature, residual_prior, estimator.name))
                    estimators.append(estimator)
    if args.add_twopl_ensembles:
        for strategy, n_groups, temperature, residual_prior, grouped_name in grouped_specs:
            for twopl_weight in twopl_weight_grid:
                for logit_bias in twopl_ensemble_bias_grid:
                    grouped_member = build_grouped_estimator(strategy, n_groups, temperature, residual_prior)
                    twopl_member = TwoPLIRTOnlineEstimator(prior_var=args.prior_var, lr=args.lr, n_fit_steps=args.n_fit_steps)
                    twopl_member.name = "twopl_irt_online_member"
                    grouped_weight = 1.0 - twopl_weight
                    twopl_pct = int(round(twopl_weight * 100))
                    grouped_pct = int(round(grouped_weight * 100))
                    bias_tag = ""
                    if abs(logit_bias) > 1e-12:
                        sign = "p" if logit_bias > 0 else "m"
                        bias_tag = f"_b{sign}{int(round(abs(logit_bias) * 1000)):03d}"
                    ensemble = WeightedAveragedEnsembleEstimator(
                        members=[grouped_member, twopl_member],
                        weights=[grouped_weight, twopl_weight],
                        name=f"{grouped_name}_twopl_hybrid_w{grouped_pct}_{twopl_pct}{bias_tag}",
                        logit_bias=float(logit_bias),
                    )
                    estimators.append(ensemble)
    if args.add_rasch_baselines:
        rasch = RaschIRTOnlineEstimator(prior_var=args.prior_var, lr=args.lr, n_fit_steps=args.n_fit_steps)
        rasch.name = "rasch_irt_online_baseline"
        twopl = TwoPLIRTOnlineEstimator(prior_var=args.prior_var, lr=args.lr, n_fit_steps=args.n_fit_steps)
        twopl.name = "twopl_irt_online_baseline"
        estimators.extend([rasch, twopl])

    print(
        f"Running grouped residual IRT sweep: models={len(estimators)} "
        f"strategies={strategy_grid} groups={n_groups_grid} "
        f"temps={temperature_grid} residual_priors={residual_prior_grid} "
        f"add_twopl_ensembles={args.add_twopl_ensembles}"
    )
    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name="responses_static",
        data_mode=f"grouped_residual_irt_{args.query_sequence}_{args.feature_set}",
        splits=splits,
        estimators=estimators,
        policies=[UniformRandomPolicy()],
        budgets=budgets,
        rng=np.random.default_rng(args.seed),
        fixed_query_sequence=query_sequence,
        max_candidate_words_per_user=None,
    )
    out["query_policy"] = "fixed_global_sequence"

    summary = (
        out.groupby(["estimator", "q"], as_index=False)[
            ["balanced_accuracy", "accuracy", "nll", "brier", "auroc", "runtime_seconds"]
        ]
        .mean()
        .sort_values(["q", "balanced_accuracy"], ascending=[True, False])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "grouped_residual_irt_all.csv"
    summary_path = args.out_dir / "grouped_residual_irt_summary.csv"
    out.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    for q in sorted(set(budgets)):
        q_view = summary[summary["q"] == q].head(10).copy()
        if q_view.empty:
            continue
        q_path = args.out_dir / f"grouped_residual_irt_top_q{q}.csv"
        q_view.to_csv(q_path, index=False)
        print(f"Top models at q={q} saved to {q_path}")
    print(f"Saved raw rows to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
