from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import Estimator, UserState


class GraphLabelPropagationEstimator(Estimator):
    name = "graph_label_propagation"

    def __init__(self, n_neighbors: int, alpha: float, n_iter: int) -> None:
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_iter = n_iter
        self.neighbors = np.empty((0, 0), dtype=np.int32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        if n_words == 0:
            self.neighbors = np.empty((0, 0), dtype=np.int32)
            return
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, n_words), metric="cosine")
        nn.fit(word_features)
        idx = nn.kneighbors(return_distance=False)
        self.neighbors = idx[:, 1:].astype(np.int32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"obs": {}, "cached": None})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        obs = dict(user_state.payload["obs"])
        for w, y in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            obs[int(w)] = float(y)
        return UserState(payload={"obs": obs, "cached": None})

    def _compute_prob(self, user_state: UserState) -> np.ndarray:
        n_words = self.neighbors.shape[0]
        p = np.full(n_words, 0.5, dtype=np.float32)
        obs = user_state.payload["obs"]
        for w, y in obs.items():
            p[w] = y
        observed = set(obs.keys())
        observed_mask = np.zeros(n_words, dtype=bool)
        if observed:
            observed_mask[np.array(list(observed), dtype=np.int32)] = True
        for _ in range(self.n_iter):
            new_p = self.alpha * p[self.neighbors].mean(axis=1) + (1.0 - self.alpha) * 0.5
            new_p[observed_mask] = p[observed_mask]
            p = np.clip(new_p, 1e-6, 1 - 1e-6)
        return p

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if user_state.payload.get("cached") is None:
            full = self._compute_prob(user_state)
        else:
            full = user_state.payload["cached"]
        return np.clip(full[candidate_word_ids], 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)
