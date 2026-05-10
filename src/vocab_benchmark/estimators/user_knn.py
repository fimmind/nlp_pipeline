from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Estimator, UserState


class UserKNNResponseEstimator(Estimator):
    name = "user_knn_response"

    def __init__(self, n_neighbors: int, prior_blend: float, similarity_temperature: float) -> None:
        self.n_neighbors = n_neighbors
        self.prior_blend = prior_blend
        self.similarity_temperature = similarity_temperature
        self.word_prior = np.zeros(0, dtype=np.float32)
        self.user_word_matrix = np.empty((0, 0), dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = int(word_features.shape[0])
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if train_responses.empty:
            self.user_word_matrix = np.empty((0, n_words), dtype=np.float32)
            return

        users = sorted(train_responses["user_idx"].astype(int).unique().tolist())
        user_to_row = {user_id: row_idx for row_idx, user_id in enumerate(users)}
        matrix = np.full((len(users), n_words), np.nan, dtype=np.float32)
        for row in train_responses[["user_idx", "word_idx", "label"]].itertuples(index=False):
            matrix[user_to_row[int(row.user_idx)], int(row.word_idx)] = float(row.label)

        counts = np.sum(~np.isnan(matrix), axis=0)
        sums = np.nansum(matrix, axis=0)
        prior = np.divide(sums + 1.0, counts + 2.0, out=np.full(n_words, 0.5, dtype=np.float32), where=counts >= 0)
        self.word_prior = np.clip(prior.astype(np.float32), 1e-5, 1 - 1e-5)
        self.user_word_matrix = matrix

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _neighbor_weights(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.user_word_matrix.shape[0] == 0 or len(observed_word_ids) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

        train_observed = self.user_word_matrix[:, observed_word_ids]
        valid = ~np.isnan(train_observed)
        centered_train = np.where(valid, train_observed - self.word_prior[observed_word_ids].reshape(1, -1), 0.0)
        centered_user = (observed_labels - self.word_prior[observed_word_ids]).reshape(1, -1)
        numerator = np.sum(centered_train * centered_user * valid, axis=1)
        train_norm = np.sqrt(np.sum((centered_train * valid) ** 2, axis=1))
        user_norm = float(np.sqrt(np.sum(centered_user**2)))
        denom = train_norm * max(user_norm, 1e-8)
        similarity = np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=np.float32), where=denom > 1e-8)
        k = min(self.n_neighbors, len(similarity))
        if k <= 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        neighbor_ids = np.argpartition(-similarity, kth=k - 1)[:k]
        neighbor_ids = neighbor_ids[np.argsort(-similarity[neighbor_ids])]
        selected_similarity = similarity[neighbor_ids]
        scaled = selected_similarity / max(self.similarity_temperature, 1e-6)
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled).astype(np.float32)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-8:
            weights = np.full(k, 1.0 / k, dtype=np.float32)
        else:
            weights = (weights / weights_sum).astype(np.float32)
        return neighbor_ids.astype(np.int32), weights

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        neighbor_ids, weights = self._neighbor_weights(observed_word_ids, observed_labels)
        prior = self.word_prior[candidate_word_ids]
        if len(neighbor_ids) == 0:
            return np.clip(prior.astype(np.float32), 1e-6, 1 - 1e-6)

        neighbor_labels = self.user_word_matrix[neighbor_ids][:, candidate_word_ids]
        valid = ~np.isnan(neighbor_labels)
        weighted_sum = np.nansum(np.where(valid, neighbor_labels, 0.0) * weights.reshape(-1, 1), axis=0)
        weight_sum = np.sum(valid * weights.reshape(-1, 1), axis=0)
        collaborative = np.divide(weighted_sum, weight_sum, out=prior.copy(), where=weight_sum > 1e-8)
        p = self.prior_blend * prior + (1.0 - self.prior_blend) * collaborative
        return np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
