from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .base import Estimator, UserState


class UserSimilarityKNNOnlineEstimator(Estimator):
    name = "user_similarity_knn_online"

    def __init__(
        self,
        n_neighbors: int,
        shrinkage: float,
        prior_weight: float,
        temperature: float,
        logit_bias: float,
        rate_centering_weight: float,
        rate_alpha: float,
        rate_beta: float,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.prior_weight = prior_weight
        self.temperature = temperature
        self.logit_bias = logit_bias
        self.rate_centering_weight = rate_centering_weight
        self.rate_alpha = rate_alpha
        self.rate_beta = rate_beta
        self.user_matrix = np.zeros((0, 0), dtype=np.float32)
        self.word_prior = np.zeros(0, dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = int(word_features.shape[0])
        users = sorted(train_responses["user_idx"].astype(int).unique().tolist()) if not train_responses.empty else []
        user_to_row = {u: i for i, u in enumerate(users)}
        self.user_matrix = np.full((len(users), n_words), np.nan, dtype=np.float32)
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if train_responses.empty:
            return
        for row in train_responses[["user_idx", "word_idx", "label"]].itertuples(index=False):
            self.user_matrix[user_to_row[int(row.user_idx)], int(row.word_idx)] = float(row.label)
        counts = np.sum(~np.isnan(self.user_matrix), axis=0)
        sums = np.nansum(self.user_matrix, axis=0)
        prior = np.divide(sums, counts, out=np.full(n_words, 0.5, dtype=np.float32), where=counts > 0)
        self.word_prior = np.clip(prior.astype(np.float32), 1e-5, 1 - 1e-5)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed": {}})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        observed = dict(user_state.payload["observed"])
        for word_id, label in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            observed[int(word_id)] = float(label)
        return UserState(payload={"observed": observed})

    def _similar_users(self, observed: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        if len(observed) == 0 or self.user_matrix.shape[0] == 0:
            idx = np.arange(self.user_matrix.shape[0], dtype=np.int32)
            weights = np.ones(len(idx), dtype=np.float32) / max(1, len(idx))
            return idx, weights
        word_ids = np.array(sorted(observed.keys()), dtype=np.int32)
        labels = np.array([observed[int(w)] for w in word_ids], dtype=np.float32)
        train = self.user_matrix[:, word_ids]
        valid = ~np.isnan(train)
        centered_labels = labels - 0.5
        train_centered = np.where(valid, train - 0.5, 0.0)
        numerator = train_centered @ centered_labels
        denom = np.sqrt(np.sum(train_centered * train_centered, axis=1) * float(np.sum(centered_labels * centered_labels)) + 1e-8)
        sim = numerator / denom
        coverage = valid.mean(axis=1)
        sim = sim * coverage * (len(word_ids) / (len(word_ids) + self.shrinkage))
        if len(sim) <= self.n_neighbors:
            idx = np.arange(len(sim), dtype=np.int32)
        else:
            idx = np.argpartition(-sim, kth=self.n_neighbors - 1)[: self.n_neighbors].astype(np.int32)
        selected = sim[idx]
        selected = np.maximum(selected, 0.0)
        if float(selected.sum()) <= 1e-8:
            selected = np.ones(len(idx), dtype=np.float32)
        weights = selected.astype(np.float32) / float(selected.sum())
        return idx, weights

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        observed = user_state.payload["observed"]
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.user_matrix.shape[0] == 0:
            return self.word_prior[candidate_word_ids]
        idx, weights = self._similar_users(observed)
        neighbor_labels = self.user_matrix[idx][:, candidate_word_ids]
        weighted = np.nansum(neighbor_labels * weights.reshape(-1, 1), axis=0)
        observed_weight = np.nansum((~np.isnan(neighbor_labels)).astype(np.float32) * weights.reshape(-1, 1), axis=0)
        collab = np.divide(weighted, observed_weight, out=self.word_prior[candidate_word_ids].copy(), where=observed_weight > 1e-8)
        prior = self.word_prior[candidate_word_ids]
        collaborative_logit = logit(np.clip(collab, 1e-5, 1 - 1e-5)) / self.temperature
        prior_logit = logit(prior)
        score = (1.0 - self.prior_weight) * collaborative_logit + self.prior_weight * prior_logit + self.logit_bias
        if len(observed) > 0 and self.rate_centering_weight > 0.0:
            observed_word_ids = np.array(list(observed.keys()), dtype=np.int32)
            observed_labels = np.array(list(observed.values()), dtype=np.float32)
            observed_rate = (float(observed_labels.sum()) + self.rate_alpha) / (
                float(len(observed_labels)) + self.rate_alpha + self.rate_beta
            )
            expected_rate = float(np.mean(self.word_prior[observed_word_ids]))
            rate_delta = logit(np.clip(observed_rate, 1e-5, 1 - 1e-5)) - logit(np.clip(expected_rate, 1e-5, 1 - 1e-5))
            score = score + self.rate_centering_weight * rate_delta
        p = expit(score)
        return np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
