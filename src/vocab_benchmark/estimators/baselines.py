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

