from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import Estimator, UserState


class OnlineUserLogisticEstimator(Estimator):
    name = "online_user_logistic"

    def __init__(self, regularization_c: float, prior_blend: float, class_weight_balanced: bool, min_observations: int) -> None:
        self.regularization_c = regularization_c
        self.prior_blend = prior_blend
        self.class_weight_balanced = class_weight_balanced
        self.min_observations = min_observations
        self.word_features = np.empty((0, 0), dtype=np.float32)
        self.feature_mean = np.empty(0, dtype=np.float32)
        self.feature_std = np.empty(0, dtype=np.float32)
        self.word_prior = np.zeros(0, dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        features = word_features.astype(np.float32)
        self.feature_mean = features.mean(axis=0).astype(np.float32)
        self.feature_std = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
        self.word_features = ((features - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(np.float32)
        n_words = int(features.shape[0])
        prior = np.full(n_words, 0.5, dtype=np.float32)
        if not train_responses.empty:
            grouped = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grouped.iterrows():
                prior[int(idx)] = float((row["sum"] + 1.0) / (row["count"] + 2.0))
        self.word_prior = np.clip(prior, 1e-5, 1 - 1e-5)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _fit_user_model(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> LogisticRegression | None:
        if len(observed_word_ids) < self.min_observations or len(np.unique(observed_labels)) < 2:
            return None
        class_weight = "balanced" if self.class_weight_balanced else None
        model = LogisticRegression(
            C=self.regularization_c,
            class_weight=class_weight,
            max_iter=300,
            solver="lbfgs",
        )
        model.fit(self.word_features[observed_word_ids], observed_labels.astype(np.int32))
        return model

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        prior = self.word_prior[candidate_word_ids]
        model = self._fit_user_model(observed_word_ids, observed_labels)
        if model is None:
            return prior.astype(np.float32)
        user_probs = model.predict_proba(self.word_features[candidate_word_ids])[:, 1].astype(np.float32)
        probs = self.prior_blend * prior + (1.0 - self.prior_blend) * user_probs
        return np.clip(probs.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
