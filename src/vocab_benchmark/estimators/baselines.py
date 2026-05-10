from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Estimator, UserState


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


class GlobalWordPriorEstimator(Estimator):
    name = "global_word_prior"

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.word_prior = np.array([], dtype=np.float32)
        self.global_rate = 0.5

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if train_responses.empty:
            return
        grp = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
        for idx, row in grp.iterrows():
            self.word_prior[int(idx)] = (row["sum"] + self.alpha) / (row["count"] + self.alpha + self.beta)
        self.global_rate = float((train_responses["label"].sum() + self.alpha) / (len(train_responses) + self.alpha + self.beta))

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"n": 0, "s": 0.0})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        payload = dict(user_state.payload)
        payload["n"] += int(len(observed_labels))
        payload["s"] += float(observed_labels.sum())
        return UserState(payload=payload)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        return _clip(self.word_prior[candidate_word_ids])

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)


class UserRateDifficultyEstimator(Estimator):
    name = "user_rate_difficulty"

    def __init__(self, alpha: float, beta: float, blend: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.blend = blend
        self.global_prior = np.array([], dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.global_prior = np.full(n_words, 0.5, dtype=np.float32)
        if train_responses.empty:
            return
        grp = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
        for idx, row in grp.iterrows():
            self.global_prior[int(idx)] = (row["sum"] + self.alpha) / (row["count"] + self.alpha + self.beta)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"n": 0, "s": 0.0})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        payload = dict(user_state.payload)
        payload["n"] += int(len(observed_labels))
        payload["s"] += float(observed_labels.sum())
        return UserState(payload=payload)

    def _user_rate(self, payload: dict) -> float:
        return float((payload["s"] + self.alpha) / (payload["n"] + self.alpha + self.beta))

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        user_rate = self._user_rate(user_state.payload)
        centered = self.global_prior[candidate_word_ids] - 0.5
        p = user_rate + self.blend * centered
        return _clip(p)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        n = user_state.payload["n"]
        return p * (1 - p) + 1.0 / (n + 1.0)


class DifficultyStratifiedBetaEstimator(Estimator):
    name = "difficulty_stratified_beta"

    def __init__(self, alpha: float, beta: float, n_bins: int) -> None:
        self.alpha = alpha
        self.beta = beta
        self.n_bins = n_bins
        self.word_bin = np.array([], dtype=np.int32)
        self.bin_prior = np.array([], dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        base_score = word_features[:, -1] if word_features.shape[1] > 0 else np.zeros(n_words)
        qs = np.quantile(base_score, np.linspace(0, 1, self.n_bins + 1))
        bins = np.clip(np.digitize(base_score, qs[1:-1], right=True), 0, self.n_bins - 1)
        self.word_bin = bins.astype(np.int32)
        self.bin_prior = np.full(self.n_bins, 0.5, dtype=np.float32)
        if train_responses.empty:
            return
        tmp = train_responses.assign(bin_id=self.word_bin[train_responses["word_idx"].to_numpy()])
        grp = tmp.groupby("bin_id")["label"].agg(["sum", "count"])
        for idx, row in grp.iterrows():
            self.bin_prior[int(idx)] = (row["sum"] + self.alpha) / (row["count"] + self.alpha + self.beta)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"sum": np.zeros(self.n_bins), "count": np.zeros(self.n_bins)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        s = user_state.payload["sum"].copy()
        c = user_state.payload["count"].copy()
        bins = self.word_bin[observed_word_ids]
        for b, y in zip(bins.tolist(), observed_labels.tolist()):
            s[b] += y
            c[b] += 1
        return UserState(payload={"sum": s, "count": c})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        bins = self.word_bin[candidate_word_ids]
        s = user_state.payload["sum"][bins]
        c = user_state.payload["count"][bins]
        post = (s + self.alpha) / (c + self.alpha + self.beta)
        p = 0.5 * post + 0.5 * self.bin_prior[bins]
        return _clip(p)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        bins = self.word_bin[candidate_word_ids]
        c = user_state.payload["count"][bins]
        return 1.0 / (c + self.alpha + self.beta + 1.0)
