from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .base import Estimator, UserState


class PersonalizedKNNPriorEstimator(Estimator):
    name = "personalized_knn_prior"

    def __init__(self, n_neighbors: int, alpha: float, beta: float, prior_blend: float) -> None:
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.beta = beta
        self.prior_blend = prior_blend
        self.word_prior = np.array([], dtype=np.float32)
        self.neighbors = np.empty((0, 0), dtype=np.int32)
        self.weights = np.empty((0, 0), dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if n_words == 0:
            self.neighbors = np.empty((0, 0), dtype=np.int32)
            self.weights = np.empty((0, 0), dtype=np.float32)
            return
        if not train_responses.empty:
            grp = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grp.iterrows():
                self.word_prior[int(idx)] = float((row["sum"] + self.alpha) / (row["count"] + self.alpha + self.beta))

        sim = cosine_similarity(word_features)
        np.fill_diagonal(sim, -1.0)
        k = min(max(1, self.n_neighbors), n_words - 1 if n_words > 1 else 1)
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, axis=1)
        w = np.clip(w, 0.0, None)
        denom = w.sum(axis=1, keepdims=True)
        denom[denom <= 1e-8] = 1.0
        self.neighbors = idx.astype(np.int32)
        self.weights = (w / denom).astype(np.float32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed": {}})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        observed = dict(user_state.payload["observed"])
        for w, y in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            observed[int(w)] = float(y)
        return UserState(payload={"observed": observed})

    def _predict_one(self, observed: dict[int, float], word_id: int) -> float:
        prior = float(self.word_prior[word_id])
        if not observed or self.neighbors.shape[0] == 0:
            return prior
        nbrs = self.neighbors[word_id]
        w = self.weights[word_id]
        obs_vals = []
        obs_w = []
        for i, nbr in enumerate(nbrs.tolist()):
            if nbr in observed:
                obs_vals.append(float(observed[nbr]))
                obs_w.append(float(w[i]))
        if len(obs_vals) == 0:
            return prior
        local = float(np.dot(np.array(obs_vals, dtype=np.float32), np.array(obs_w, dtype=np.float32)) / (np.sum(obs_w) + 1e-8))
        post = self.prior_blend * prior + (1.0 - self.prior_blend) * local
        return float(np.clip(post, 1e-6, 1 - 1e-6))

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        observed = user_state.payload["observed"]
        out = np.array([self._predict_one(observed, int(w)) for w in candidate_word_ids.tolist()], dtype=np.float32)
        return np.clip(out, 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class PersonalizedHybridSignalEstimator(Estimator):
    name = "personalized_hybrid_signal"

    def __init__(
        self,
        n_neighbors: int,
        alpha: float,
        beta: float,
        prior_weight: float,
        local_weight: float,
        user_weight: float,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.beta = beta
        self.prior_weight = prior_weight
        self.local_weight = local_weight
        self.user_weight = user_weight
        self.word_prior = np.array([], dtype=np.float32)
        self.neighbors = np.empty((0, 0), dtype=np.int32)
        self.weights = np.empty((0, 0), dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if n_words == 0:
            self.neighbors = np.empty((0, 0), dtype=np.int32)
            self.weights = np.empty((0, 0), dtype=np.float32)
            return
        if not train_responses.empty:
            grp = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grp.iterrows():
                self.word_prior[int(idx)] = float((row["sum"] + self.alpha) / (row["count"] + self.alpha + self.beta))
        sim = cosine_similarity(word_features)
        np.fill_diagonal(sim, -1.0)
        k = min(max(1, self.n_neighbors), n_words - 1 if n_words > 1 else 1)
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, axis=1)
        w = np.clip(w, 0.0, None)
        denom = w.sum(axis=1, keepdims=True)
        denom[denom <= 1e-8] = 1.0
        self.neighbors = idx.astype(np.int32)
        self.weights = (w / denom).astype(np.float32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed": {}, "sum": 0.0, "count": 0})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        observed = dict(user_state.payload["observed"])
        total = float(user_state.payload["sum"])
        count = int(user_state.payload["count"])
        for w, y in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            observed[int(w)] = float(y)
            total += float(y)
            count += 1
        return UserState(payload={"observed": observed, "sum": total, "count": count})

    def _local_signal(self, observed: dict[int, float], word_id: int) -> float:
        if not observed or self.neighbors.shape[0] == 0:
            return 0.5
        nbrs = self.neighbors[word_id]
        w = self.weights[word_id]
        vals: list[float] = []
        ww: list[float] = []
        for i, nbr in enumerate(nbrs.tolist()):
            if nbr in observed:
                vals.append(float(observed[nbr]))
                ww.append(float(w[i]))
        if len(vals) == 0:
            return 0.5
        return float(np.dot(np.array(vals, dtype=np.float32), np.array(ww, dtype=np.float32)) / (np.sum(ww) + 1e-8))

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        observed = user_state.payload["observed"]
        user_rate = float((user_state.payload["sum"] + self.alpha) / (user_state.payload["count"] + self.alpha + self.beta))
        out: list[float] = []
        for word_id in candidate_word_ids.tolist():
            w = int(word_id)
            prior = float(self.word_prior[w])
            local = self._local_signal(observed, w)
            score = (
                self.prior_weight * (prior - 0.5)
                + self.local_weight * (local - 0.5)
                + self.user_weight * (user_rate - 0.5)
            )
            p = 1.0 / (1.0 + np.exp(-4.0 * score))
            out.append(float(np.clip(p, 1e-6, 1 - 1e-6)))
        return np.array(out, dtype=np.float32)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
