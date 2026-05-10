#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from vocab_benchmark.data import load_all, load_words
from vocab_benchmark.estimators.base import Estimator, UserState
from vocab_benchmark.estimators.ensemble import BudgetAdaptiveEnsembleEstimator, WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.fasttext_kernel import FastTextKernelLogisticConfig, FastTextKernelLogisticEstimator
from vocab_benchmark.estimators.irt import RaschIRTOnlineEstimator, TwoPLIRTOnlineEstimator
from vocab_benchmark.estimators.observed_user_vote import ObservedMatchUserVoteEstimator
from vocab_benchmark.estimators.online_user_logistic import OnlineUserLogisticEstimator
from vocab_benchmark.estimators.svd import SVDRidgeUserEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index


WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
PROFILE_VERSION = 1
MODEL_NAME = "budget_adaptive_refined_raw_switch500"
DEFAULT_MODEL_KEY = "best_adaptive"
MODEL_HELP: dict[str, str] = {
    "best_adaptive": "Best current practical model. Budget-adaptive Rasch/TwoPL/Vote/UserLogReg ensemble.",
    "best_high_budget": "Best high-budget fixed blend. Strong at large query budgets.",
    "rasch": "Fastest non-neural baseline: one-parameter IRT.",
    "twopl": "Fast non-neural baseline: two-parameter IRT.",
    "vote": "Fast nearest-user voting baseline.",
    "user_logreg": "Per-user logistic regression on rich word features.",
    "rasch_vote": "Fast hybrid: Rasch + vote.",
    "rasch_twopl_vote_user": "Non-neural four-way blend: Rasch + TwoPL + vote + user logreg.",
    "svd": "Fast latent collaborative non-neural model.",
    "fasttext_kernel": "Non-neural semantic kernel logistic model over fastText features.",
}
MODEL_ESTIMATED_PRECISION: dict[str, float] = {
    "best_adaptive": 0.842,
    "best_high_budget": 0.842,
    "rasch": 0.840,
    "twopl": 0.839,
    "vote": 0.833,
    "user_logreg": 0.751,
    "rasch_vote": 0.840,
    "rasch_twopl_vote_user": 0.841,
    "svd": 0.809,
    "fasttext_kernel": 0.796,
}


@dataclass(frozen=True)
class BookToken:
    raw: str
    normalized: str
    word_idx: int | None


@dataclass(frozen=True)
class BookAnalysis:
    path: Path
    token_count: int
    type_count: int
    in_vocab_token_count: int
    oov_token_count: int
    expected_unknown_token_count: int
    expected_unknown_type_count: int
    known_words: list[tuple[str, float, int]]
    unknown_words: list[tuple[str, float, int]]
    one_unknown_sentences: list[tuple[str, str, float]]


@dataclass(frozen=True)
class SentenceQueryMatch:
    sentence: str
    unknown_items: list[tuple[str, float]]
    known_token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate book vocabulary difficulty from a 100-word user profile.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--book", type=str, default=None, help="File name or path under data/example_texts.")
    parser.add_argument("--profile", type=str, default=None, help="Profile/user name used for saving and reusing answers.")
    parser.add_argument("--profile-dir", type=Path, default=Path("data/user_profiles"))
    parser.add_argument("--retake-test", action="store_true", help="Ignore an existing profile and ask the 100 questions again.")
    parser.add_argument("--answer-string", type=str, default=None, help="Automation helper: 100 chars from y/n/1/0/k/u.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY, choices=sorted(MODEL_HELP.keys()))
    parser.add_argument("--list-models", action="store_true", help="Print available models and exit.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/vocab_book_cli"))
    parser.add_argument("--disable-cache", action="store_true", help="Disable model/probability caches.")
    parser.add_argument(
        "--add-words",
        type=str,
        default=None,
        help="Add/update known/unknown labels using entries like \"apple=known,banana=unknown\".",
    )
    parser.add_argument(
        "--add-words-interactive",
        action="store_true",
        help="Interactively append or update additional known/unknown word labels in the saved profile.",
    )
    parser.add_argument(
        "--query-words",
        type=str,
        default=None,
        help="Comma-separated words to query, e.g. \"apple,banana\".",
    )
    return parser.parse_args()


def normalize_word(token: str) -> str:
    return token.lower().replace("\u2019", "'").strip("'")


def profile_path(profile_dir: Path, profile_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile_name.strip())
    if not safe:
        raise ValueError("profile must contain at least one safe filename character")
    return profile_dir / f"{safe}.json"


def build_best_estimator() -> BudgetAdaptiveEnsembleEstimator:
    low_rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
    low_rasch.name = "rasch_highbudget_var25"
    low_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
    low_vote.name = "observed_match_user_vote_t10"
    low_user_logreg = OnlineUserLogisticEstimator(
        regularization_c=0.1,
        prior_blend=0.5,
        class_weight_balanced=False,
        min_observations=50,
    )
    low_user_logreg.name = "online_user_logreg_c01_pb50"
    low_budget = WeightedAveragedEnsembleEstimator(
        members=[low_rasch, low_vote, low_user_logreg],
        weights=[0.76, 0.175, 0.065],
        name="refined_q100_raw_w760_175_065_bias_p020",
        logit_bias=0.020,
    )

    high_rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
    high_rasch.name = "rasch_highbudget_var25"
    high_twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
    high_twopl.name = "twopl_highbudget_var25"
    high_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
    high_vote.name = "observed_match_user_vote_t10"
    high_user_logreg = OnlineUserLogisticEstimator(
        regularization_c=0.1,
        prior_blend=0.5,
        class_weight_balanced=False,
        min_observations=50,
    )
    high_user_logreg.name = "online_user_logreg_c01_pb50"
    high_budget = WeightedAveragedEnsembleEstimator(
        members=[high_rasch, high_twopl, high_vote, high_user_logreg],
        weights=[0.29, 0.50, 0.13, 0.08],
        name="refined_q1000_raw_w290_500_130_080_bias_p015",
        logit_bias=0.015,
    )
    return BudgetAdaptiveEnsembleEstimator(
        low_budget=low_budget,
        high_budget=high_budget,
        switch_observations=500,
        name=MODEL_NAME,
    )


def build_estimator(model_key: str, seed: int) -> Estimator:
    if model_key == "best_adaptive":
        return build_best_estimator()
    if model_key == "best_high_budget":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(
            regularization_c=0.1,
            prior_blend=0.5,
            class_weight_balanced=False,
            min_observations=50,
        )
        user_logreg.name = "online_user_logreg_c01_pb50"
        return WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote, user_logreg],
            weights=[0.29, 0.50, 0.13, 0.08],
            name="refined_q1000_raw_w290_500_130_080_bias_p015",
            logit_bias=0.015,
        )
    if model_key == "rasch":
        estimator = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        estimator.name = "rasch_highbudget_var25"
        return estimator
    if model_key == "twopl":
        estimator = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        estimator.name = "twopl_highbudget_var25"
        return estimator
    if model_key == "vote":
        estimator = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        estimator.name = "observed_match_user_vote_t10"
        return estimator
    if model_key == "user_logreg":
        estimator = OnlineUserLogisticEstimator(
            regularization_c=0.1,
            prior_blend=0.5,
            class_weight_balanced=False,
            min_observations=50,
        )
        estimator.name = "online_user_logreg_c01_pb50"
        return estimator
    if model_key == "rasch_vote":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        return WeightedAveragedEnsembleEstimator(
            members=[rasch, vote],
            weights=[0.80, 0.20],
            name="rasch_vote_hybrid_w80_20",
            logit_bias=0.0,
        )
    if model_key == "rasch_twopl_vote_user":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(
            regularization_c=0.1,
            prior_blend=0.5,
            class_weight_balanced=False,
            min_observations=50,
        )
        user_logreg.name = "online_user_logreg_c01_pb50"
        return WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote, user_logreg],
            weights=[0.35, 0.45, 0.15, 0.05],
            name="rasch_twopl_vote_user_w35_45_15_05",
            logit_bias=0.0,
        )
    if model_key == "svd":
        estimator = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        estimator.name = "svd_ridge_r5_l1_s05"
        return estimator
    if model_key == "fasttext_kernel":
        return FastTextKernelLogisticEstimator(
            FastTextKernelLogisticConfig(
                embedding_dim=300,
                temperature=0.15,
                episodes_per_user=12,
                target_samples_per_episode=256,
                seed=seed,
                regularization_c=1.0,
                dynamic_centering_weight=1.0,
                user_rate_centering_weight=0.35,
            )
        )
    raise ValueError(f"unknown model={model_key}")


def print_model_catalog() -> None:
    print("Available models:")
    print("  (est_precision is an empirical accuracy@1000 proxy from recent benchmarks)")
    for key in sorted(MODEL_HELP.keys()):
        est = MODEL_ESTIMATED_PRECISION.get(key)
        est_text = f"{est:.3f}" if est is not None else "n/a"
        print(f"  {key:<24} est_precision={est_text}  {MODEL_HELP[key]}")


def _file_signature(path: Path) -> str:
    if not path.exists():
        return f"{path}:missing"
    stat = path.stat()
    return f"{path}:{stat.st_size}:{int(stat.st_mtime)}"


def dataset_fingerprint(data_dir: Path) -> str:
    required = [
        data_dir / "processed" / "responses_static.csv",
        data_dir / "processed" / "words.csv",
        data_dir / "processed" / "frequency.csv",
        data_dir / "processed" / "embeddings.npy",
        data_dir / "processed" / "embeddings_metadata.json",
    ]
    payload = "\n".join(_file_signature(path) for path in required)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def profile_fingerprint(observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(observed_word_ids.astype(np.int32).tobytes())
    h.update(observed_labels.astype(np.int32).tobytes())
    return h.hexdigest()


def model_cache_path(cache_dir: Path, model_key: str, data_fp: str) -> Path:
    return cache_dir / "models" / f"{model_key}_{data_fp}.pkl"


def probability_cache_path(cache_dir: Path, model_key: str, data_fp: str, profile_fp: str) -> Path:
    return cache_dir / "probabilities" / f"{model_key}_{data_fp}_{profile_fp}.npy"


def state_cache_path(cache_dir: Path, model_key: str, data_fp: str, profile_name: str) -> Path:
    safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile_name.strip())
    return cache_dir / "states" / f"{model_key}_{data_fp}_{safe_profile}.pkl"


def invalidate_probability_caches(cache_dir: Path, data_fp: str) -> int:
    probs_dir = cache_dir / "probabilities"
    if not probs_dir.exists():
        return 0
    removed = 0
    for path in probs_dir.glob(f"*_{data_fp}_*.npy"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _is_incremental_extension(
    cached_ids: np.ndarray,
    cached_labels: np.ndarray,
    current_ids: np.ndarray,
    current_labels: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    cached_map = {int(word_id): int(label) for word_id, label in zip(cached_ids.tolist(), cached_labels.tolist())}
    current_map = {int(word_id): int(label) for word_id, label in zip(current_ids.tolist(), current_labels.tolist())}
    for word_id, label in cached_map.items():
        if word_id not in current_map:
            return False, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        if int(current_map[word_id]) != int(label):
            return False, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    delta_ids: list[int] = []
    delta_labels: list[int] = []
    for word_id, label in zip(current_ids.tolist(), current_labels.tolist()):
        if int(word_id) not in cached_map:
            delta_ids.append(int(word_id))
            delta_labels.append(int(label))
    return True, np.array(delta_ids, dtype=np.int32), np.array(delta_labels, dtype=np.int32)


def load_state_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        return None
    required = {"observed_word_ids", "observed_labels", "user_state", "probs_all_words"}
    if not required.issubset(set(payload.keys())):
        return None
    return payload


def save_state_cache(
    path: Path,
    observed_word_ids: np.ndarray,
    observed_labels: np.ndarray,
    user_state: UserState,
    probs_all_words: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observed_word_ids": observed_word_ids.astype(np.int32),
        "observed_labels": observed_labels.astype(np.int32),
        "user_state": user_state,
        "probs_all_words": probs_all_words.astype(np.float32),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_state_and_probabilities(
    estimator: Estimator,
    observed_word_ids: np.ndarray,
    observed_labels: np.ndarray,
    n_words: int,
    cache_dir: Path,
    model_key: str,
    data_fp: str,
    profile_name: str,
    disable_cache: bool,
) -> tuple[UserState, np.ndarray]:
    cache_path = state_cache_path(cache_dir, model_key, data_fp, profile_name)
    state_payload = None if disable_cache else load_state_cache(cache_path)
    if state_payload is not None:
        cached_ids = np.asarray(state_payload["observed_word_ids"], dtype=np.int32)
        cached_labels = np.asarray(state_payload["observed_labels"], dtype=np.int32)
        cached_state = state_payload["user_state"]
        cached_probs = np.asarray(state_payload["probs_all_words"], dtype=np.float32)
        if cached_ids.shape == observed_word_ids.shape and cached_labels.shape == observed_labels.shape:
            if np.array_equal(cached_ids, observed_word_ids) and np.array_equal(cached_labels, observed_labels):
                if cached_probs.shape == (n_words,):
                    print(f"Loaded cached latent state and probabilities: {cache_path}")
                    return cached_state, cached_probs
        is_extension, delta_ids, delta_labels = _is_incremental_extension(
            cached_ids=cached_ids,
            cached_labels=cached_labels,
            current_ids=observed_word_ids,
            current_labels=observed_labels,
        )
        if is_extension:
            if len(delta_ids) == 0 and cached_probs.shape == (n_words,):
                print(f"Loaded cached latent state and probabilities: {cache_path}")
                return cached_state, cached_probs
            updated_state = estimator.update_user_state(cached_state, delta_ids, delta_labels)
            probs = estimator.predict_proba(updated_state, np.arange(n_words, dtype=np.int32)).astype(np.float32)
            if probs.shape != (n_words,):
                raise ValueError(f"invalid probability shape: expected {(n_words,)}, got {probs.shape}")
            if not disable_cache:
                save_state_cache(cache_path, observed_word_ids, observed_labels, updated_state, probs)
                print(f"Incrementally updated latent state cache: {cache_path}")
            return updated_state, probs
        print("Detected profile edits that changed existing labels; rebuilding latent state from scratch.")
    state = estimator.initialize_user_state()
    state = estimator.update_user_state(state, observed_word_ids, observed_labels)
    probs = estimator.predict_proba(state, np.arange(n_words, dtype=np.int32)).astype(np.float32)
    if probs.shape != (n_words,):
        raise ValueError(f"invalid probability shape: expected {(n_words,)}, got {probs.shape}")
    if not disable_cache:
        save_state_cache(cache_path, observed_word_ids, observed_labels, state, probs)
        print(f"Saved latent state cache: {cache_path}")
    return state, probs


def load_or_fit_estimator(
    model_key: str,
    seed: int,
    response_frame: pd.DataFrame | None,
    x_words: np.ndarray | None,
    cache_dir: Path,
    data_fp: str,
    disable_cache: bool,
) -> Estimator:
    cache_path = model_cache_path(cache_dir, model_key, data_fp)
    if not disable_cache and cache_path.exists():
        with cache_path.open("rb") as f:
            estimator = pickle.load(f)
        print(f"Loaded cached fitted model: {cache_path}")
        return estimator
    if response_frame is None or x_words is None:
        raise ValueError("response_frame and x_words are required when fitting a non-cached model")
    estimator = build_estimator(model_key, seed)
    estimator.fit(response_frame, x_words)
    if not disable_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved cached fitted model: {cache_path}")
    return estimator


def load_or_compute_probabilities(
    estimator: Estimator,
    state: UserState,
    n_words: int,
    cache_dir: Path,
    model_key: str,
    data_fp: str,
    profile_fp: str,
    disable_cache: bool,
) -> np.ndarray:
    cache_path = probability_cache_path(cache_dir, model_key, data_fp, profile_fp)
    if not disable_cache and cache_path.exists():
        probs = np.load(cache_path)
        if probs.shape == (n_words,):
            print(f"Loaded cached probabilities: {cache_path}")
            return probs.astype(np.float32)
    candidate_word_ids = np.arange(n_words, dtype=np.int32)
    probs = estimator.predict_proba(state, candidate_word_ids).astype(np.float32)
    if probs.shape != (n_words,):
        raise ValueError(f"invalid probability shape: expected {(n_words,)}, got {probs.shape}")
    if not disable_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, probs)
        print(f"Saved cached probabilities: {cache_path}")
    return probs


def load_word_context(data_dir: Path) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[int, str]]:
    words = load_words(data_dir).copy()
    word_index = build_word_index(words)
    lower_to_idx: dict[str, int] = {}
    idx_to_word: dict[int, str] = {}
    for idx, row in enumerate(words[["word", "word_id"]].itertuples(index=False)):
        normalized = normalize_word(str(row.word))
        if normalized and normalized not in lower_to_idx:
            lower_to_idx[normalized] = word_index[str(row.word_id)]
        idx_to_word[idx] = str(row.word)
    return words, word_index, lower_to_idx, idx_to_word


def load_full_model_context(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    loaded = load_all(data_dir)
    words = loaded.words.copy()
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency, feature_set="rich")
    user_index = {u: i for i, u in enumerate(sorted(loaded.responses_static["user_id"].astype(str).unique().tolist()))}
    word_index = build_word_index(words)
    response_frame = build_response_frame(loaded.responses_static, user_index, word_index)
    return words, x_words, response_frame


def build_query_words(response_frame: pd.DataFrame, sequence_len: int, seed: int) -> np.ndarray:
    users = sorted(response_frame["user_idx"].astype(int).unique().tolist())
    words = sorted(response_frame["word_idx"].astype(int).unique().tolist())
    if len(words) <= sequence_len:
        return np.array(words, dtype=np.int32)
    user_to_row = {user_id: row_idx for row_idx, user_id in enumerate(users)}
    word_to_col = {word_id: col_idx for col_idx, word_id in enumerate(words)}
    label_matrix = np.full((len(users), len(words)), np.nan, dtype=np.float32)
    for row in response_frame[["user_idx", "word_idx", "label"]].itertuples(index=False):
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
    base_score = base_score + rng.uniform(0.0, 1e-8, size=len(words)).astype(np.float32)

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


def parse_answer_char(answer: str) -> int:
    value = answer.strip().lower()
    if value in {"y", "yes", "1", "k", "known"}:
        return 1
    if value in {"n", "no", "0", "u", "unknown"}:
        return 0
    raise ValueError(f"invalid answer={answer!r}; use y/n, 1/0, known/unknown")


def collect_answers_interactively(words: pd.DataFrame, query_word_idx: np.ndarray) -> np.ndarray:
    labels: list[int] = []
    print("Mark each word as known or unknown. Accepted answers: y/n, 1/0, known/unknown.\n")
    print("=== 100-Word Vocabulary Test ===")
    for position, word_idx in enumerate(query_word_idx.tolist(), start=1):
        word = str(words.iloc[int(word_idx)]["word"])
        print(f"{position:3d}. {word}")
    print("")
    for position, word_idx in enumerate(query_word_idx.tolist(), start=1):
        word = str(words.iloc[int(word_idx)]["word"])
        while True:
            raw = input(f"{position:3d}/100  {word}: ").strip()
            try:
                labels.append(parse_answer_char(raw))
                break
            except ValueError as exc:
                print(exc)
    return np.array(labels, dtype=np.int32)


def collect_answers_from_string(answer_string: str, expected_len: int) -> np.ndarray:
    compact = re.sub(r"[\s,;|]+", "", answer_string)
    if len(compact) != expected_len:
        raise ValueError(f"answer-string must contain exactly {expected_len} answers after removing separators; got {len(compact)}")
    return np.array([parse_answer_char(ch) for ch in compact], dtype=np.int32)


def parse_additional_entries(
    add_words: str,
    lower_to_idx: dict[str, int],
    idx_to_word: dict[int, str],
) -> list[tuple[int, int]]:
    entries: list[tuple[int, int]] = []
    parts = [part.strip() for part in add_words.split(",") if part.strip()]
    if len(parts) == 0:
        return entries
    for part in parts:
        if "=" not in part:
            raise ValueError(f"invalid add-words entry: {part!r}; expected word=label")
        word_raw, label_raw = part.split("=", 1)
        normalized = normalize_word(word_raw.strip())
        if normalized not in lower_to_idx:
            raise ValueError(f"word not in model vocabulary: {word_raw.strip()!r}")
        word_idx = int(lower_to_idx[normalized])
        label = parse_answer_char(label_raw.strip())
        entries.append((word_idx, label))
    return deduplicate_entries_preserving_input_order(entries)


def collect_additional_entries_interactively(
    lower_to_idx: dict[str, int],
    idx_to_word: dict[int, str],
) -> list[tuple[int, int]]:
    print("Enter additional words to label as known/unknown. Empty word finishes input.")
    print("Labels accepted: y/n, known/unknown, 1/0.")
    entries: list[tuple[int, int]] = []
    while True:
        raw_word = input("word: ").strip()
        if raw_word == "":
            break
        normalized = normalize_word(raw_word)
        if normalized not in lower_to_idx:
            print(f"Word {raw_word!r} is not in model vocabulary. Try another word.")
            continue
        word_idx = int(lower_to_idx[normalized])
        canonical_word = idx_to_word[word_idx]
        raw_label = input(f"label for {canonical_word} (known/unknown): ").strip()
        try:
            label = parse_answer_char(raw_label)
        except ValueError as exc:
            print(exc)
            continue
        entries.append((word_idx, label))
    return deduplicate_entries_preserving_input_order(entries)


def deduplicate_entries_preserving_input_order(entries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(entries) == 0:
        return []
    # Keep only the last assignment for each word, while preserving the
    # relative order of those last assignments as entered by the user.
    last_pos_by_word: dict[int, int] = {}
    label_by_word: dict[int, int] = {}
    for pos, (word_idx, label) in enumerate(entries):
        last_pos_by_word[int(word_idx)] = pos
        label_by_word[int(word_idx)] = int(label)
    ordered_words = sorted(last_pos_by_word.items(), key=lambda item: item[1])
    return [(word_idx, label_by_word[word_idx]) for word_idx, _ in ordered_words]


def parse_query_words(query_words: str) -> list[str]:
    return [normalize_word(part) for part in query_words.split(",") if normalize_word(part) != ""]


def apply_additional_entries(
    observed_word_ids: np.ndarray,
    observed_labels: np.ndarray,
    extra_entries: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(extra_entries) == 0:
        return observed_word_ids, observed_labels, 0
    pos_by_word: dict[int, int] = {int(word_idx): idx for idx, word_idx in enumerate(observed_word_ids.tolist())}
    ids = observed_word_ids.astype(np.int32).copy()
    labels = observed_labels.astype(np.int32).copy()
    changed = 0
    for word_idx, label in extra_entries:
        existing_pos = pos_by_word.get(int(word_idx))
        if existing_pos is None:
            ids = np.concatenate([ids, np.array([word_idx], dtype=np.int32)])
            labels = np.concatenate([labels, np.array([label], dtype=np.int32)])
            pos_by_word[int(word_idx)] = len(ids) - 1
            changed += 1
            continue
        if int(labels[existing_pos]) != int(label):
            labels[existing_pos] = int(label)
            changed += 1
    return ids, labels, changed


def save_profile(path: Path, profile_name: str, words: pd.DataFrame, query_word_idx: np.ndarray, labels: np.ndarray) -> None:
    if len(query_word_idx) != len(labels):
        raise ValueError("query_word_idx and labels must have same length")
    entries: list[dict[str, Any]] = []
    for word_idx, label in zip(query_word_idx.tolist(), labels.tolist()):
        row = words.iloc[int(word_idx)]
        entries.append({"word_id": str(row["word_id"]), "word": str(row["word"]), "label": int(label)})
    payload = {
        "version": PROFILE_VERSION,
        "profile": profile_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "profile_model_agnostic",
        "feature_set": "rich",
        "query_count": int(len(entries)),
        "answers": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_profile(path: Path, word_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != PROFILE_VERSION:
        raise ValueError(f"unsupported profile version in {path}")
    entries = payload.get("answers", [])
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"profile has no answers: {path}")
    word_ids: list[int] = []
    labels: list[int] = []
    for entry in entries:
        word_id = str(entry["word_id"])
        if word_id not in word_index:
            raise ValueError(f"profile word_id is not in current vocabulary: {word_id}")
        word_ids.append(word_index[word_id])
        labels.append(int(entry["label"]))
    return np.array(word_ids, dtype=np.int32), np.array(labels, dtype=np.int32), payload


def select_book(example_dir: Path, requested: str | None) -> Path:
    if requested is not None:
        candidate = Path(requested)
        if not candidate.exists():
            candidate = example_dir / requested
        if not candidate.exists():
            raise FileNotFoundError(f"book not found: {requested}")
        return candidate
    books = sorted(example_dir.glob("*.txt"))
    if len(books) == 0:
        raise FileNotFoundError(f"no .txt books found under {example_dir}")
    print("Available books:")
    for idx, path in enumerate(books, start=1):
        print(f"  {idx}. {path.name}")
    while True:
        raw = input("Select a book number: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a numeric book index.")
            continue
        if 1 <= choice <= len(books):
            return books[choice - 1]
        print(f"Enter a number from 1 to {len(books)}.")


def tokenize_book(text: str, lower_to_idx: dict[str, int]) -> list[BookToken]:
    tokens: list[BookToken] = []
    for match in WORD_RE.finditer(text):
        raw = match.group(0)
        normalized = normalize_word(raw)
        tokens.append(BookToken(raw=raw, normalized=normalized, word_idx=lower_to_idx.get(normalized)))
    return tokens


def split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]


def predict_for_book_words(probs_all_words: np.ndarray, word_indices: np.ndarray) -> dict[int, float]:
    if len(word_indices) == 0:
        return {}
    probs = probs_all_words[word_indices.astype(np.int32)]
    return {int(word_idx): float(prob) for word_idx, prob in zip(word_indices.tolist(), probs.tolist())}


def analyze_book(
    path: Path,
    lower_to_idx: dict[str, int],
    idx_to_word: dict[int, str],
    probs_all_words: np.ndarray,
    threshold: float,
    seed: int,
) -> BookAnalysis:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize_book(text, lower_to_idx)
    token_counts: dict[str, int] = {}
    word_to_idx: dict[str, int] = {}
    oov_token_count = 0
    for token in tokens:
        token_counts[token.normalized] = token_counts.get(token.normalized, 0) + 1
        if token.word_idx is None:
            oov_token_count += 1
        elif token.normalized not in word_to_idx:
            word_to_idx[token.normalized] = int(token.word_idx)
    unique_indices = np.array(sorted(set(word_to_idx.values())), dtype=np.int32)
    probabilities = predict_for_book_words(probs_all_words, unique_indices)

    expected_unknown_token_count = 0
    expected_unknown_types: set[str] = {word for word, idx in word_to_idx.items() if probabilities[int(idx)] < threshold}
    for word in expected_unknown_types:
        expected_unknown_token_count += token_counts[word]

    known_rows: list[tuple[str, float, int]] = []
    unknown_rows: list[tuple[str, float, int]] = []
    for word, idx in word_to_idx.items():
        probability = probabilities[int(idx)]
        row = (word, probability, token_counts[word])
        if probability >= threshold:
            known_rows.append(row)
        else:
            unknown_rows.append(row)
    rng = np.random.default_rng(seed)
    sampled_known = sample_rows(known_rows, 25, rng)
    sampled_unknown = sample_rows(unknown_rows, 25, rng)
    sentence_rows = find_one_unknown_sentences(text, lower_to_idx, idx_to_word, probabilities, threshold, seed)
    return BookAnalysis(
        path=path,
        token_count=len(tokens),
        type_count=len(token_counts),
        in_vocab_token_count=len(tokens) - oov_token_count,
        oov_token_count=oov_token_count,
        expected_unknown_token_count=expected_unknown_token_count,
        expected_unknown_type_count=len(expected_unknown_types),
        known_words=sampled_known,
        unknown_words=sampled_unknown,
        one_unknown_sentences=sentence_rows,
    )


def sample_rows(rows: list[tuple[str, float, int]], sample_size: int, rng: np.random.Generator) -> list[tuple[str, float, int]]:
    sorted_rows = sorted(rows, key=lambda item: item[0])
    if len(sorted_rows) <= sample_size:
        return sorted_rows
    selected = rng.choice(np.arange(len(sorted_rows)), size=sample_size, replace=False)
    return [sorted_rows[int(idx)] for idx in np.sort(selected).tolist()]


def find_one_unknown_sentences(
    text: str,
    lower_to_idx: dict[str, int],
    idx_to_word: dict[int, str],
    probabilities: dict[int, float],
    threshold: float,
    seed: int,
) -> list[tuple[str, str, float]]:
    candidates: list[tuple[str, str, float]] = []
    for sentence in split_sentences(text):
        raw_tokens = [match.group(0) for match in WORD_RE.finditer(sentence)]
        if len(raw_tokens) < 4 or len(raw_tokens) > 35:
            continue
        unknown_words: list[tuple[str, float]] = []
        has_oov = False
        for raw in raw_tokens:
            normalized = normalize_word(raw)
            word_idx = lower_to_idx.get(normalized)
            if word_idx is None or int(word_idx) not in probabilities:
                has_oov = True
                break
            probability = probabilities[int(word_idx)]
            if probability < threshold:
                unknown_words.append((idx_to_word[int(word_idx)], probability))
        if not has_oov and len(unknown_words) == 1:
            candidates.append((sentence, unknown_words[0][0], unknown_words[0][1]))
    rng = np.random.default_rng(seed + 1009)
    if len(candidates) <= 10:
        return candidates
    selected = rng.choice(np.arange(len(candidates)), size=10, replace=False)
    return [candidates[int(idx)] for idx in np.sort(selected).tolist()]


def print_analysis(analysis: BookAnalysis) -> None:
    unknown_pct = 0.0 if analysis.in_vocab_token_count == 0 else 100.0 * analysis.expected_unknown_token_count / analysis.in_vocab_token_count
    oov_pct = 0.0 if analysis.token_count == 0 else 100.0 * analysis.oov_token_count / analysis.token_count
    print("\n=== Book Vocabulary Estimate ===")
    print(f"Book: {analysis.path.name}")
    print(f"Word tokens analyzed: {analysis.token_count}")
    print(f"Unique word types: {analysis.type_count}")
    print(f"In-vocabulary tokens used for estimates: {analysis.in_vocab_token_count}")
    print(f"Out-of-model-vocabulary tokens discarded from estimates: {analysis.oov_token_count} ({oov_pct:.2f}% of all tokens)")
    print(f"Estimated unknown in-vocabulary tokens: {analysis.expected_unknown_token_count} ({unknown_pct:.2f}% of in-vocabulary tokens)")
    print(f"Estimated unknown in-vocabulary unique types: {analysis.expected_unknown_type_count}")

    print("\n=== Random 25 Words Expected Known ===")
    print_word_rows(analysis.known_words)
    print("\n=== Random 25 Words Expected Unknown ===")
    print_word_rows(analysis.unknown_words)

    print("\n=== Sentences Expected To Have Exactly One Unknown Word ===")
    if len(analysis.one_unknown_sentences) == 0:
        print("No matching sentences found under the strict criterion: all model-vocabulary words known except one unknown.")
    for idx, (sentence, unknown_word, probability) in enumerate(analysis.one_unknown_sentences, start=1):
        print(f"{idx}. [{unknown_word}, p_known={probability:.3f}] {sentence}")


def print_word_rows(rows: list[tuple[str, float, int]]) -> None:
    if len(rows) == 0:
        print("No words found in this category.")
        return
    for word, probability, count in rows:
        print(f"  {word:<24} p_known={probability:.3f}  count={count}")


def build_last_observed_map(observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for pos, (word_idx, label) in enumerate(zip(observed_word_ids.tolist(), observed_labels.tolist()), start=1):
        out[int(word_idx)] = (int(label), pos)
    return out


def find_query_word_sentences(
    query_word: str,
    book_text: str,
    lower_to_idx: dict[str, int],
    probs_all_words: np.ndarray,
    threshold: float,
) -> list[SentenceQueryMatch]:
    rows: list[SentenceQueryMatch] = []
    for sentence in split_sentences(book_text):
        raw_tokens = [match.group(0) for match in WORD_RE.finditer(sentence)]
        normalized_tokens = [normalize_word(token) for token in raw_tokens]
        if query_word not in normalized_tokens:
            continue
        unknown_items: list[tuple[str, float]] = []
        known_count = 0
        seen_unknown: set[str] = set()
        for token in normalized_tokens:
            word_idx = lower_to_idx.get(token)
            if word_idx is None:
                continue
            probability = float(probs_all_words[int(word_idx)])
            if probability < threshold:
                if token not in seen_unknown:
                    unknown_items.append((token, probability))
                    seen_unknown.add(token)
            else:
                known_count += 1
        rows.append(SentenceQueryMatch(sentence=sentence, unknown_items=unknown_items, known_token_count=known_count))
    rows.sort(key=lambda row: (len(row.unknown_items), -row.known_token_count, row.sentence))
    return rows


def print_query_word_report(
    query_words: list[str],
    lower_to_idx: dict[str, int],
    idx_to_word: dict[int, str],
    probs_all_words: np.ndarray,
    observed_word_ids: np.ndarray,
    observed_labels: np.ndarray,
    threshold: float,
    book_path: Path | None,
) -> None:
    if len(query_words) == 0:
        return
    observed_map = build_last_observed_map(observed_word_ids, observed_labels)
    book_text: str | None = None
    if book_path is not None:
        book_text = book_path.read_text(encoding="utf-8", errors="ignore")
    print("\n=== Query Words ===")
    for word in query_words:
        print(f"\nWord: {word}")
        word_idx = lower_to_idx.get(word)
        if word_idx is None:
            print("  Not in model vocabulary.")
            continue
        canonical_word = idx_to_word[int(word_idx)]
        probability = float(probs_all_words[int(word_idx)])
        predicted_label = "known" if probability >= threshold else "unknown"
        print(f"  Model prediction: p_known={probability:.3f} -> {predicted_label}")
        if int(word_idx) in observed_map:
            observed_label, observed_pos = observed_map[int(word_idx)]
            observed_text = "known" if observed_label == 1 else "unknown"
            print(f"  Observed before: yes (last observed state={observed_text}, observation_index={observed_pos})")
        else:
            print("  Observed before: no")
        if book_text is not None:
            rows = find_query_word_sentences(
                query_word=word,
                book_text=book_text,
                lower_to_idx=lower_to_idx,
                probs_all_words=probs_all_words,
                threshold=threshold,
            )
            print(f"  Sentences in book sorted by predicted unknown-word count (least -> most): {len(rows)}")
            if len(rows) == 0:
                print("  No matching sentences found.")
            for idx, row in enumerate(rows, start=1):
                unknown_desc = ", ".join([f"{unknown_word}:{prob:.3f}" for unknown_word, prob in row.unknown_items])
                if unknown_desc == "":
                    unknown_desc = "none"
                print(f"    {idx}. unknown_count={len(row.unknown_items)} known_token_count={row.known_token_count} unknown_words=[{unknown_desc}]")
                print(f"       {row.sentence}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_model_catalog()
        return
    if args.profile is None or args.profile.strip() == "":
        raise ValueError("--profile is required unless --list-models is used")
    if args.known_threshold <= 0.0 or args.known_threshold >= 1.0:
        raise ValueError("known-threshold must be between 0 and 1")

    data_fp = dataset_fingerprint(args.data_dir)
    words, word_index, lower_to_idx, idx_to_word = load_word_context(args.data_dir)
    path = profile_path(args.profile_dir, args.profile)
    profile_changed = False
    if path.exists() and not args.retake_test:
        observed_word_ids, observed_labels, payload = load_profile(path, word_index)
        print(f"Loaded profile: {path} ({len(observed_labels)} answers, created_at={payload.get('created_at')})")
    else:
        _words_full, _x_words, response_frame = load_full_model_context(args.data_dir)
        query_word_idx = build_query_words(response_frame, 100, args.seed)
        if len(query_word_idx) < 100:
            raise ValueError(f"query sequence produced only {len(query_word_idx)} words")
        if args.answer_string is None:
            observed_labels = collect_answers_interactively(words, query_word_idx)
        else:
            observed_labels = collect_answers_from_string(args.answer_string, 100)
        observed_word_ids = query_word_idx.astype(np.int32)
        save_profile(path, args.profile, words, observed_word_ids, observed_labels)
        print(f"Saved profile: {path}")
        profile_changed = True

    extra_entries: list[tuple[int, int]] = []
    if args.add_words is not None:
        extra_entries.extend(parse_additional_entries(args.add_words, lower_to_idx, idx_to_word))
    if args.add_words_interactive:
        extra_entries.extend(collect_additional_entries_interactively(lower_to_idx, idx_to_word))
    if len(extra_entries) > 0:
        observed_word_ids, observed_labels, changed_count = apply_additional_entries(
            observed_word_ids=observed_word_ids,
            observed_labels=observed_labels,
            extra_entries=extra_entries,
        )
        if changed_count > 0:
            save_profile(path, args.profile, words, observed_word_ids, observed_labels)
            profile_changed = True
            print(f"Updated profile with {changed_count} additional/updated labels: {path}")
        else:
            print("Additional labels did not change the profile.")

    if profile_changed and not args.disable_cache:
        removed_count = invalidate_probability_caches(args.cache_dir, data_fp)
        if removed_count > 0:
            print(f"Invalidated {removed_count} stale probability cache file(s) for dataset fingerprint {data_fp}.")

    n_words = len(words)
    model_path = model_cache_path(args.cache_dir, args.model, data_fp)
    if not args.disable_cache and model_path.exists():
        estimator = load_or_fit_estimator(
            model_key=args.model,
            seed=args.seed,
            response_frame=None,
            x_words=None,
            cache_dir=args.cache_dir,
            data_fp=data_fp,
            disable_cache=args.disable_cache,
        )
    else:
        words_full, x_words, response_frame = load_full_model_context(args.data_dir)
        if len(words_full) != len(words):
            raise ValueError("word table mismatch between lightweight and full context")
        estimator = load_or_fit_estimator(
            model_key=args.model,
            seed=args.seed,
            response_frame=response_frame,
            x_words=x_words,
            cache_dir=args.cache_dir,
            data_fp=data_fp,
            disable_cache=args.disable_cache,
        )
        n_words = x_words.shape[0]
    print(f"Using model: {args.model} ({estimator.name})")
    state, probs_all_words = build_state_and_probabilities(
        estimator=estimator,
        observed_word_ids=observed_word_ids,
        observed_labels=observed_labels,
        n_words=n_words,
        cache_dir=args.cache_dir,
        model_key=args.model,
        data_fp=data_fp,
        profile_name=args.profile,
        disable_cache=args.disable_cache,
    )

    query_words = parse_query_words(args.query_words) if args.query_words is not None else []

    book_path: Path | None = None
    if args.book is not None:
        book_path = select_book(args.data_dir / "example_texts", args.book)
        if len(query_words) == 0:
            analysis = analyze_book(
                path=book_path,
                lower_to_idx=lower_to_idx,
                idx_to_word=idx_to_word,
                probs_all_words=probs_all_words,
                threshold=args.known_threshold,
                seed=args.seed,
            )
            print_analysis(analysis)
    if len(query_words) > 0:
        print_query_word_report(
            query_words=query_words,
            lower_to_idx=lower_to_idx,
            idx_to_word=idx_to_word,
            probs_all_words=probs_all_words,
            observed_word_ids=observed_word_ids,
            observed_labels=observed_labels,
            threshold=args.known_threshold,
            book_path=book_path,
        )


if __name__ == "__main__":
    main()
