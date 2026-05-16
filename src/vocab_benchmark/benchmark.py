from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_all
from .estimators.baselines import GlobalWordPriorEstimator
from .estimators.irt import RaschIRTOnlineEstimator, TwoPLIRTOnlineEstimator
from .estimators.neural import AveragedEnsembleEstimator, NeuralEncoderDecoderEstimator, NeuralEstimatorConfig
from .features import build_response_frame, build_word_feature_matrix, build_word_index
from .metrics import classification_metrics
from .query_policies import DifficultyStratifiedRandomPolicy, EmbeddingKMedoidsPolicy, EntropyPolicy, UniformRandomPolicy
from .synthetic import generate_synthetic_dataset


def _build_loou_splits(user_ids: list[str]) -> list[dict[str, Any]]:
    return [{"split_id": f"loou_{u}", "test_user_id": u} for u in user_ids]


def _extract_split_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "splits" in payload:
        rows = payload["splits"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        test_user = row.get("test_user_id", row.get("test_user"))
        if test_user is None:
            continue
        out.append({"split_id": row.get("split_id", f"split_{i}"), "test_user_id": str(test_user)})
    return out


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    for (est, pol), g in df.groupby(["estimator", "query_policy"]):
        agg = g.groupby("q")[metric].mean().reset_index()
        plt.plot(agg["q"], agg[metric], marker="o", label=f"{est}|{pol}")
    plt.xlabel("q")
    plt.ylabel(metric)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _build_fixed_query_sequence(resp: pd.DataFrame, x_words: np.ndarray, sequence_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grp = resp.groupby("word_idx")
    observed_counts = grp["label"].count().to_dict()
    user_coverage = grp["user_idx"].nunique().to_dict()
    label_mean = grp["label"].mean().to_dict()
    candidate_words = np.array(sorted(observed_counts.keys()), dtype=np.int32)
    if len(candidate_words) <= sequence_len:
        return candidate_words
    difficulty = x_words[:, -1] if x_words.shape[1] > 0 else np.zeros(x_words.shape[0], dtype=np.float32)
    scores = []
    for w in candidate_words.tolist():
        p = float(label_mean.get(int(w), 0.5))
        entropy_proxy = -abs(p - 0.5)
        scores.append(
            (
                int(w),
                float(difficulty[w]),
                float(observed_counts.get(int(w), 0)),
                float(user_coverage.get(int(w), 0)),
                float(entropy_proxy),
            )
        )
    tmp = pd.DataFrame(scores, columns=["word_idx", "difficulty", "count", "user_coverage", "entropy_proxy"])
    tmp["diff_bin"] = pd.qcut(tmp["difficulty"], q=10, labels=False, duplicates="drop")
    chosen: list[int] = []
    bins = sorted(tmp["diff_bin"].dropna().astype(int).unique().tolist())
    per_bin = max(1, sequence_len // max(1, len(bins)))
    for b in bins:
        pool = tmp[tmp["diff_bin"] == b].copy()
        pool = pool.sort_values(["user_coverage", "entropy_proxy", "count"], ascending=[False, False, False])
        # Deterministic tie-breaking with seeded shuffle inside equal-score bands.
        pool["_rnd"] = np.random.default_rng(seed + b).uniform(size=len(pool))
        pool = pool.sort_values(["user_coverage", "entropy_proxy", "count", "_rnd"], ascending=[False, False, False, True])
        top = pool.head(per_bin)["word_idx"].astype(int).tolist()
        chosen.extend(top)
    if len(chosen) < sequence_len:
        remaining = [int(w) for w in candidate_words.tolist() if int(w) not in set(chosen)]
        rem = tmp[tmp["word_idx"].isin(remaining)].copy()
        rem = rem.sort_values(["user_coverage", "entropy_proxy", "count"], ascending=[False, False, False])
        remaining = rem["word_idx"].astype(int).tolist()
        chosen.extend(remaining)
    return np.array(chosen[:sequence_len], dtype=np.int32)


def _build_user_discriminative_query_sequence(resp: pd.DataFrame, sequence_len: int, seed: int) -> np.ndarray:
    users = sorted(resp["user_idx"].astype(int).unique().tolist())
    words = sorted(resp["word_idx"].astype(int).unique().tolist())
    if len(words) <= sequence_len:
        return np.array(words, dtype=np.int32)
    user_to_row = {user_id: row_idx for row_idx, user_id in enumerate(users)}
    word_to_col = {word_id: col_idx for col_idx, word_id in enumerate(words)}
    label_matrix = np.full((len(users), len(words)), np.nan, dtype=np.float32)
    for row in resp[["user_idx", "word_idx", "label"]].itertuples(index=False):
        label_matrix[user_to_row[int(row.user_idx)], word_to_col[int(row.word_idx)]] = float(row.label)

    valid = ~np.isnan(label_matrix)
    coverage = valid.mean(axis=0)
    means = np.divide(np.nansum(label_matrix, axis=0), np.maximum(valid.sum(axis=0), 1), dtype=np.float32)
    centered = np.where(valid, label_matrix - means.reshape(1, -1), 0.0).astype(np.float32)
    norms = np.linalg.norm(centered, axis=0)
    variance = norms * norms / np.maximum(valid.sum(axis=0), 1)
    entropy_proxy = 1.0 - np.abs(means - 0.5) * 2.0
    base_score = variance * np.maximum(coverage, 1e-6) * np.maximum(entropy_proxy, 1e-6)
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(0.0, 1e-8, size=len(words)).astype(np.float32)
    base_score = base_score + tie_break
    if sequence_len > 200:
        candidate_pool_size = min(len(words), max(sequence_len * 4, 3000))
        if candidate_pool_size < len(words):
            pool_cols = np.argpartition(-base_score, kth=candidate_pool_size - 1)[:candidate_pool_size]
            pool_cols = pool_cols[np.argsort(-base_score[pool_cols])]
            words = [int(words[col]) for col in pool_cols.tolist()]
            centered = centered[:, pool_cols]
            norms = norms[pool_cols]
            base_score = base_score[pool_cols]
    selected_cols: list[int] = []
    selected_unit = np.zeros((0, len(users)), dtype=np.float32)
    available = np.ones(len(words), dtype=bool)
    unit_vectors = (centered / np.maximum(norms.reshape(1, -1), 1e-8)).T
    for _ in range(min(sequence_len, len(words))):
        if len(selected_cols) == 0:
            adjusted = np.where(available, base_score, -np.inf)
        else:
            redundancy = np.max(np.abs(unit_vectors @ selected_unit.T), axis=1)
            adjusted = np.where(available, base_score * (1.0 - 0.75 * redundancy), -np.inf)
        col = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[col]):
            break
        selected_cols.append(col)
        available[col] = False
        selected_unit = unit_vectors[np.array(selected_cols, dtype=np.int32)]
    selected_words = [int(words[col]) for col in selected_cols]
    if len(selected_words) < sequence_len:
        remaining_cols = np.where(available)[0]
        order = remaining_cols[np.argsort(-base_score[remaining_cols])]
        selected_words.extend([int(words[col]) for col in order[: sequence_len - len(selected_words)].tolist()])
    return np.array(selected_words[:sequence_len], dtype=np.int32)


def _build_prior_uncertain_query_sequence(resp: pd.DataFrame, sequence_len: int, seed: int) -> np.ndarray:
    grouped = resp.groupby("word_idx")["label"].agg(["mean", "count"])
    if grouped.empty:
        return np.zeros(0, dtype=np.int32)
    tmp = grouped.reset_index()
    tmp["entropy_proxy"] = 1.0 - np.abs(tmp["mean"] - 0.5) * 2.0
    rng = np.random.default_rng(seed)
    tmp["tie_break"] = rng.uniform(0.0, 1e-8, size=len(tmp))
    tmp = tmp.sort_values(["entropy_proxy", "count", "tie_break"], ascending=[False, False, True])
    return tmp["word_idx"].head(sequence_len).to_numpy(dtype=np.int32)


def _evaluate_responses(
    responses: pd.DataFrame,
    words: pd.DataFrame,
    x_words: np.ndarray,
    embedding_backend: str,
    dataset_name: str,
    data_mode: str,
    splits: list[dict[str, Any]],
    estimators: list[Any],
    policies: list[Any],
    budgets: list[int],
    rng: np.random.Generator,
    fixed_query_sequence: np.ndarray | None = None,
    cold_word_idx: set[int] | None = None,
    max_interactions_per_user: int | None = None,
    max_candidate_words_per_user: int | None = None,
    evaluation_protocol: str = "loou_user_generalization",
) -> pd.DataFrame:
    users = sorted(responses["user_id"].astype(str).unique().tolist())
    user_index = {u: i for i, u in enumerate(users)}
    word_index = build_word_index(words)
    resp = build_response_frame(responses, user_index, word_index)
    if max_interactions_per_user is not None and max_interactions_per_user > 0:
        resp = (
            resp.sort_values(["user_idx"])
            .groupby("user_idx", group_keys=False)
            .head(max_interactions_per_user)
            .reset_index(drop=True)
        )
    all_rows: list[dict[str, Any]] = []
    for estimator in estimators:
        for policy in policies:
            for split_i, split in enumerate(splits):
                test_user = str(split["test_user_id"])
                if test_user not in user_index:
                    continue
                test_uidx = user_index[test_user]
                user_rows = resp[resp["user_idx"] == test_uidx]
                if user_rows.empty:
                    continue
                train_rows = resp[resp["user_idx"] != test_uidx]
                if cold_word_idx is not None:
                    train_rows = train_rows[~train_rows["word_idx"].isin(cold_word_idx)]
                estimator.fit(train_rows, x_words)
                state = estimator.initialize_user_state()
                labels_by_word = {int(r.word_idx): int(r.label) for r in user_rows.itertuples()}
                candidate = np.array(sorted(labels_by_word.keys()), dtype=np.int32)
                if max_candidate_words_per_user is not None and len(candidate) > max_candidate_words_per_user:
                    candidate = np.sort(rng.choice(candidate, size=max_candidate_words_per_user, replace=False))
                queried: set[int] = set()
                observed_ids = np.array([], dtype=np.int32)
                observed_labels = np.array([], dtype=np.int32)
                h0 = None
                queries_to_confidence_80 = np.nan
                start = time.perf_counter()
                for q in budgets:
                    if q > len(candidate):
                        continue
                    needed = q - len(queried)
                    if needed > 0:
                        if fixed_query_sequence is None:
                            new_q = policy.select_next_queries(estimator, state, candidate, queried, needed, rng)
                        else:
                            allowed = [int(w) for w in fixed_query_sequence.tolist() if int(w) in set(candidate.tolist()) and int(w) not in queried]
                            new_q = np.array(allowed[:needed], dtype=np.int32)
                        new_l = np.array([labels_by_word[int(w)] for w in new_q], dtype=np.int32)
                        queried.update(new_q.tolist())
                        observed_ids = np.concatenate([observed_ids, new_q])
                        observed_labels = np.concatenate([observed_labels, new_l])
                        state = estimator.update_user_state(state, new_q, new_l)
                    eval_ids = np.array([w for w in candidate.tolist() if w not in queried], dtype=np.int32)
                    if len(eval_ids) == 0:
                        continue
                    y_true = np.array([labels_by_word[int(w)] for w in eval_ids], dtype=np.int32)
                    y_prob = estimator.predict_proba(state, eval_ids)
                    m = classification_metrics(y_true, y_prob)
                    if h0 is None:
                        h0 = m["mean_predictive_entropy"]
                    nur = 0.0 if h0 <= 1e-12 else (h0 - m["mean_predictive_entropy"]) / h0
                    if np.isnan(queries_to_confidence_80) and m["confident_fraction_0_2_0_8"] >= 0.8:
                        queries_to_confidence_80 = float(q)
                    row = {
                        "dataset_name": dataset_name,
                        "data_mode": data_mode,
                        "evaluation_protocol": evaluation_protocol,
                        "embedding_backend": embedding_backend,
                        "estimator": estimator.name,
                        "query_policy": policy.name,
                        "split_id": split.get("split_id", f"split_{split_i}"),
                        "user_id": test_user,
                        "q": q,
                        "n_train_users": int(len(train_rows["user_idx"].unique())),
                        "n_words": int(len(words)),
                        "n_observed_labels": int(len(observed_ids)),
                        "n_eval_labels": int(len(eval_ids)),
                        "normalized_uncertainty_reduction": float(nur),
                        "queries_to_confidence_80": queries_to_confidence_80,
                        "runtime_seconds": float(time.perf_counter() - start),
                    }
                    row.update(m)
                    all_rows.append(row)
    return pd.DataFrame(all_rows)


def _evaluate_within_user_completion(
    responses: pd.DataFrame,
    words: pd.DataFrame,
    x_words: np.ndarray,
    embedding_backend: str,
    dataset_name: str,
    data_mode: str,
    estimators: list[Any],
    policies: list[Any],
    budgets: list[int],
    rng: np.random.Generator,
    fixed_query_sequence: np.ndarray | None = None,
    max_interactions_per_user: int | None = None,
    max_candidate_words_per_user: int | None = None,
    n_repeats: int = 3,
    evaluation_protocol: str = "within_user_completion",
) -> pd.DataFrame:
    users = sorted(responses["user_id"].astype(str).unique().tolist())
    user_index = {u: i for i, u in enumerate(users)}
    word_index = build_word_index(words)
    resp = build_response_frame(responses, user_index, word_index)
    if max_interactions_per_user is not None and max_interactions_per_user > 0:
        resp = (
            resp.sort_values(["user_idx"])
            .groupby("user_idx", group_keys=False)
            .head(max_interactions_per_user)
            .reset_index(drop=True)
        )
    all_rows: list[dict[str, Any]] = []
    for estimator in estimators:
        estimator.fit(resp, x_words)
        for policy in policies:
            for repeat_idx in range(n_repeats):
                for test_user in users:
                    test_uidx = user_index[test_user]
                    user_rows = resp[resp["user_idx"] == test_uidx]
                    if user_rows.empty:
                        continue
                    state = estimator.initialize_user_state()
                    labels_by_word = {int(r.word_idx): int(r.label) for r in user_rows.itertuples()}
                    candidate = np.array(sorted(labels_by_word.keys()), dtype=np.int32)
                    if max_candidate_words_per_user is not None and len(candidate) > max_candidate_words_per_user:
                        candidate = np.sort(rng.choice(candidate, size=max_candidate_words_per_user, replace=False))
                    queried: set[int] = set()
                    observed_ids = np.array([], dtype=np.int32)
                    observed_labels = np.array([], dtype=np.int32)
                    h0 = None
                    queries_to_confidence_80 = np.nan
                    start = time.perf_counter()
                    for q in budgets:
                        if q > len(candidate):
                            continue
                        needed = q - len(queried)
                        if needed > 0:
                            if fixed_query_sequence is None:
                                new_q = policy.select_next_queries(estimator, state, candidate, queried, needed, rng)
                            else:
                                allowed = [
                                    int(w)
                                    for w in fixed_query_sequence.tolist()
                                    if int(w) in set(candidate.tolist()) and int(w) not in queried
                                ]
                                new_q = np.array(allowed[:needed], dtype=np.int32)
                            new_l = np.array([labels_by_word[int(w)] for w in new_q], dtype=np.int32)
                            queried.update(new_q.tolist())
                            observed_ids = np.concatenate([observed_ids, new_q])
                            observed_labels = np.concatenate([observed_labels, new_l])
                            state = estimator.update_user_state(state, new_q, new_l)
                        eval_ids = np.array([w for w in candidate.tolist() if w not in queried], dtype=np.int32)
                        if len(eval_ids) == 0:
                            continue
                        y_true = np.array([labels_by_word[int(w)] for w in eval_ids], dtype=np.int32)
                        y_prob = estimator.predict_proba(state, eval_ids)
                        m = classification_metrics(y_true, y_prob)
                        if h0 is None:
                            h0 = m["mean_predictive_entropy"]
                        nur = 0.0 if h0 <= 1e-12 else (h0 - m["mean_predictive_entropy"]) / h0
                        if np.isnan(queries_to_confidence_80) and m["confident_fraction_0_2_0_8"] >= 0.8:
                            queries_to_confidence_80 = float(q)
                        row = {
                            "dataset_name": dataset_name,
                            "data_mode": data_mode,
                            "evaluation_protocol": evaluation_protocol,
                            "embedding_backend": embedding_backend,
                            "estimator": estimator.name,
                            "query_policy": policy.name,
                            "split_id": f"within_user_{repeat_idx}_{test_user}",
                            "user_id": test_user,
                            "q": q,
                            "n_train_users": int(len(users)),
                            "n_words": int(len(words)),
                            "n_observed_labels": int(len(observed_ids)),
                            "n_eval_labels": int(len(eval_ids)),
                            "normalized_uncertainty_reduction": float(nur),
                            "queries_to_confidence_80": queries_to_confidence_80,
                            "runtime_seconds": float(time.perf_counter() - start),
                        }
                        row.update(m)
                        all_rows.append(row)
    return pd.DataFrame(all_rows)


def _fit_neural_candidates(seed: int) -> list[NeuralEstimatorConfig]:
    arch = ["gru_mlp", "lstm_bilinear", "residual_gru_gated"]
    strategies = ["teacher_forced", "curriculum_prefix", "contrastive_hard_negative"]
    hidden = [64, 128]
    lrs = [1e-3, 3e-4]
    dropout = [0.0, 0.2]
    out: list[NeuralEstimatorConfig] = []
    for a in arch:
        for s in strategies:
            for h in hidden:
                for lr in lrs:
                    for d in dropout:
                        out.append(
                            NeuralEstimatorConfig(
                                architecture=a,
                                strategy=s,
                                hidden_dim=h,
                                lr=lr,
                                dropout=d,
                                max_epochs=20,
                                seed=seed,
                                weight_decay=1e-4,
                                calibration_weight=0.02,
                                early_stopping_patience=4,
                            )
                        )
    return out


def _select_neural_estimators_by_validation(
    resp: pd.DataFrame,
    words: pd.DataFrame,
    x_words: np.ndarray,
    splits: list[dict[str, Any]],
    fixed_query_sequence: np.ndarray,
    embedding_backend: str,
    dataset_name: str,
    seed: int,
) -> tuple[list[NeuralEncoderDecoderEstimator], list[dict[str, float]]]:
    candidates = _fit_neural_candidates(seed=seed)
    rng = np.random.default_rng(seed)
    scored: list[tuple[float, float, float, NeuralEncoderDecoderEstimator]] = []
    summaries: list[dict[str, float]] = []
    for cfg in candidates:
        est = NeuralEncoderDecoderEstimator(cfg)
        eval_df = _evaluate_responses(
            responses=resp,
            words=words,
            x_words=x_words,
            embedding_backend=embedding_backend,
            dataset_name=dataset_name,
            data_mode="static_neural_validation",
            splits=splits,
            estimators=[est],
            policies=[UniformRandomPolicy()],
            budgets=[50, 100],
            rng=rng,
            fixed_query_sequence=fixed_query_sequence,
        )
        if eval_df.empty:
            continue
        ba = float(eval_df["balanced_accuracy"].mean())
        nll = float(eval_df["nll"].mean())
        runtime = float(eval_df["runtime_seconds"].mean())
        scored.append((ba, -nll, -runtime, est))
        summaries.append({"balanced_accuracy": ba, "nll": nll, "runtime_seconds": runtime, "estimator": est.name})
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = [row[3] for row in scored[:3]]
    return selected, summaries


def _queries_to_target(summary_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["estimator"]
    if "evaluation_protocol" in summary_df.columns:
        group_cols = ["evaluation_protocol", "estimator"]
    for group_key, g in summary_df.groupby(group_cols):
        tmp = g.sort_values("q")
        if isinstance(group_key, tuple):
            evaluation_protocol = str(group_key[0])
            estimator_name = str(group_key[1])
        else:
            evaluation_protocol = "unknown"
            estimator_name = str(group_key)
        for threshold in thresholds:
            reached = tmp[tmp["balanced_accuracy"] >= threshold]
            q = int(reached["q"].min()) if not reached.empty else np.nan
            rows.append(
                {
                    "evaluation_protocol": evaluation_protocol,
                    "estimator": estimator_name,
                    "target_balanced_accuracy": threshold,
                    "queries_to_target": q,
                }
            )
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines = [header, separator]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join([str(v) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def run_benchmark(data_dir: Path, reports_dir: Path, seed: int, max_users: int) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    loaded = load_all(data_dir)
    synthetic_mode = loaded.words.empty or loaded.responses_static.empty
    if synthetic_mode:
        syn = generate_synthetic_dataset(n_words=300, n_users=80, embed_dim=32, rng=rng)
        words = syn.words
        frequency = syn.frequency
        embeddings = syn.embeddings
        responses_static = syn.responses_static
        embedding_backend = "synthetic_hash_like"
        dataset_name = "responses_static"
        responses_temporal = loaded.responses_temporal.copy()
    else:
        words = loaded.words
        frequency = loaded.frequency
        embeddings = loaded.embeddings
        responses_static = loaded.responses_static
        embedding_backend = loaded.embedding_backend
        dataset_name = "responses_static"
        responses_temporal = loaded.responses_temporal.copy()
    users = sorted(responses_static["user_id"].astype(str).unique().tolist())
    if max_users > 0:
        keep = set(users[:max_users])
        responses_static = responses_static[responses_static["user_id"].astype(str).isin(keep)].copy()
    x_words = build_word_feature_matrix(words, embeddings, frequency)
    word_index = build_word_index(words)
    if synthetic_mode or "static_leave_one_user_out" not in loaded.splits:
        splits = _build_loou_splits(sorted(responses_static["user_id"].astype(str).unique().tolist()))
    else:
        splits = _extract_split_rows(loaded.splits["static_leave_one_user_out"])
    validation_users = set()
    if not synthetic_mode and "static_validation_users" in loaded.splits:
        validation_users = set(map(str, loaded.splits["static_validation_users"].get("user_ids", [])))
    if not validation_users:
        split_users = sorted({str(s["test_user_id"]) for s in splits})
        validation_users = set(split_users[: max(1, len(split_users) // 5)])
    validation_splits = [s for s in splits if str(s["test_user_id"]) in validation_users]
    test_splits = [s for s in splits if str(s["test_user_id"]) not in validation_users]
    if not test_splits:
        test_splits = splits
    difficulty = x_words[:, -1] if x_words.shape[1] > 0 else np.zeros(len(x_words))
    estimators = [
        GlobalWordPriorEstimator(alpha=1.0, beta=1.0),
        RaschIRTOnlineEstimator(prior_var=4.0, lr=1.0, n_fit_steps=3),
        TwoPLIRTOnlineEstimator(prior_var=4.0, lr=1.0, n_fit_steps=3),
    ]
    policies = [
        UniformRandomPolicy(),
        DifficultyStratifiedRandomPolicy(difficulty=difficulty, n_bins=5),
        EntropyPolicy(),
        EmbeddingKMedoidsPolicy(features=embeddings if embeddings.shape[0] > 0 else x_words, n_clusters=min(50, max(1, len(words))), random_state=seed),
    ]
    static_budgets = [0, 1, 2, 5, 10, 20, 50, 100]
    fixed_budgets = [50, 100, 200]
    temporal_budgets = [0, 1, 2, 5, 10, 20, 50, 100]

    static_df = _evaluate_within_user_completion(
        responses=responses_static,
        words=words,
        x_words=x_words,
        embedding_backend=embedding_backend,
        dataset_name=dataset_name,
        data_mode="synthetic_within_user" if synthetic_mode else "static_within_user",
        estimators=estimators,
        policies=policies,
        budgets=static_budgets,
        rng=rng,
        n_repeats=3,
        max_candidate_words_per_user=2000,
    )

    resp_frame = build_response_frame(responses_static, {u: i for i, u in enumerate(sorted(responses_static["user_id"].astype(str).unique().tolist()))}, word_index)
    fixed_query_sequence = _build_fixed_query_sequence(resp=resp_frame, x_words=x_words, sequence_len=200, seed=seed)

    neural_rows = pd.DataFrame()
    if not responses_static.empty and len(fixed_query_sequence) > 0:
        selected_neural, selection_summary = _select_neural_estimators_by_validation(
            resp=responses_static,
            words=words,
            x_words=x_words,
            splits=validation_splits,
            fixed_query_sequence=fixed_query_sequence,
            embedding_backend=embedding_backend,
            dataset_name=dataset_name,
            seed=seed,
        )
        neural_estimators: list[Any] = []
        for est in selected_neural:
            for model_seed in [seed, seed + 1, seed + 2]:
                cfg = est.config
                neural_estimators.append(
                    NeuralEncoderDecoderEstimator(
                        NeuralEstimatorConfig(
                            architecture=cfg.architecture,
                            strategy=cfg.strategy,
                            hidden_dim=cfg.hidden_dim,
                            lr=cfg.lr,
                            dropout=cfg.dropout,
                            max_epochs=cfg.max_epochs,
                            seed=model_seed,
                            weight_decay=cfg.weight_decay,
                            calibration_weight=cfg.calibration_weight,
                            early_stopping_patience=cfg.early_stopping_patience,
                        )
                    )
                )
        if len(neural_estimators) >= 2:
            ensemble = AveragedEnsembleEstimator(neural_estimators[:3], name="neural_ensemble_top3", temperature=1.0)
            neural_estimators.append(ensemble)
        neural_rows = _evaluate_responses(
            responses=responses_static,
            words=words,
            x_words=x_words,
            embedding_backend=embedding_backend,
            dataset_name=dataset_name,
            data_mode="static_fixed200_neural",
            splits=test_splits,
            estimators=neural_estimators,
            policies=[UniformRandomPolicy()],
            budgets=fixed_budgets,
            rng=rng,
            fixed_query_sequence=fixed_query_sequence,
            max_candidate_words_per_user=2000,
        )
        if not neural_rows.empty:
            neural_rows["query_policy"] = "fixed_global_200"
            selection_df = pd.DataFrame(selection_summary)
            if not selection_df.empty:
                selection_df.to_csv(reports_dir / "neural_validation_selection.csv", index=False)

    cold_word_df = pd.DataFrame()
    if not synthetic_mode and "cold_word_split" in loaded.splits and not responses_static.empty:
        cold_ids = set(map(str, loaded.splits["cold_word_split"].get("cold_word_ids", [])))
        cold_word_idx = {word_index[w] for w in cold_ids if w in word_index}
        cold_word_df = _evaluate_responses(
            responses=responses_static,
            words=words,
            x_words=x_words,
            embedding_backend=embedding_backend,
            dataset_name=dataset_name,
            data_mode="cold_word",
            splits=splits,
            estimators=estimators,
            policies=policies,
            budgets=static_budgets,
            rng=rng,
            cold_word_idx=cold_word_idx,
            max_candidate_words_per_user=2000,
        )

    temporal_df = pd.DataFrame()
    if not responses_temporal.empty:
        temporal = responses_temporal.dropna(subset=["user_id", "word_id", "label"]).copy()
        temporal["label"] = temporal["label"].astype(int)
        temporal = temporal[temporal["word_id"].astype(str).isin(set(words["word_id"].astype(str).tolist()))]
        if max_users > 0:
            keep_temporal = set(sorted(temporal["user_id"].astype(str).unique().tolist())[:max_users])
            temporal = temporal[temporal["user_id"].astype(str).isin(keep_temporal)].copy()
        temporal_splits = _build_loou_splits(sorted(temporal["user_id"].astype(str).unique().tolist()))
        temporal_df = _evaluate_responses(
            responses=temporal.rename(columns={"source": "source"}),
            words=words,
            x_words=x_words,
            embedding_backend=embedding_backend,
            dataset_name="responses_temporal",
            data_mode="temporal",
            splits=temporal_splits,
            estimators=estimators,
            policies=policies,
            budgets=temporal_budgets,
            rng=rng,
            max_interactions_per_user=1500,
            max_candidate_words_per_user=1000,
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots = reports_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    all_static_df = pd.concat([static_df, neural_rows], ignore_index=True) if not neural_rows.empty else static_df
    static_path = reports_dir / ("results_synthetic.csv" if synthetic_mode else "results_static.csv")
    all_static_df.to_csv(static_path, index=False)
    cold_path = reports_dir / "results_cold_word.csv"
    temporal_path = reports_dir / "results_temporal.csv"
    cold_word_df.to_csv(cold_path, index=False)
    temporal_df.to_csv(temporal_path, index=False)

    if not all_static_df.empty:
        leaderboard_df = (
            all_static_df[all_static_df["q"].isin([50, 100, 200])]
            .groupby(["evaluation_protocol", "estimator", "q"], as_index=False)[
                ["balanced_accuracy", "accuracy", "nll", "brier", "auroc"]
            ]
            .mean()
        )
        leaderboard_path = reports_dir / "leaderboard_static_fixed200.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        qt = _queries_to_target(leaderboard_df, thresholds=[0.90, 0.95])
        qt_path = reports_dir / "queries_to_target.csv"
        qt.to_csv(qt_path, index=False)
        merged = leaderboard_df.merge(qt, on=["evaluation_protocol", "estimator"], how="left")
        (reports_dir / "leaderboard_static_fixed200.md").write_text(_to_markdown_table(merged), encoding="utf-8")
        within_user_rows = leaderboard_df[leaderboard_df["evaluation_protocol"] == "within_user_completion"]
        best_source = within_user_rows if not within_user_rows.empty else leaderboard_df
        best = best_source[best_source["q"] <= 200]["balanced_accuracy"].max()
        statement = {
            "evaluation_protocol": "within_user_completion" if not within_user_rows.empty else "mixed",
            "reached_balanced_accuracy_0_95_at_q_le_200": bool(best >= 0.95 if pd.notna(best) else False),
        }
        (reports_dir / "balanced_accuracy_target_statement.json").write_text(json.dumps(statement, indent=2), encoding="utf-8")

    _plot_metric(all_static_df, "nll", plots / "nll_vs_q.png")
    _plot_metric(all_static_df, "brier", plots / "brier_vs_q.png")
    _plot_metric(all_static_df, "auroc", plots / "auroc_vs_q.png")
    _plot_metric(all_static_df, "mean_predictive_entropy", plots / "entropy_vs_q.png")
    _plot_metric(all_static_df, "normalized_uncertainty_reduction", plots / "normalized_uncertainty_reduction_vs_q.png")
    _plot_metric(all_static_df, "uncertain_fraction_0_4_0_6", plots / "uncertain_fraction_vs_q.png")
    _plot_metric(all_static_df, "confident_fraction_0_2_0_8", plots / "confident_fraction_vs_q.png")
    return {"results_static": static_path, "results_cold_word": cold_path, "results_temporal": temporal_path}
