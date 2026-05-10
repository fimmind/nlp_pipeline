from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Estimator, UserState


class ObservedMatchUserVoteEstimator(Estimator):
    name = "observed_match_user_vote"

    def __init__(self, temperature: float, prior_blend: float, power: float) -> None:
        self.temperature = temperature
        self.prior_blend = prior_blend
        self.power = power
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

    def _weights(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> np.ndarray:
        n_users = int(self.user_word_matrix.shape[0])
        if n_users == 0:
            return np.zeros(0, dtype=np.float32)
        if len(observed_word_ids) == 0:
            return np.full(n_users, 1.0 / n_users, dtype=np.float32)
        train_observed = self.user_word_matrix[:, observed_word_ids]
        valid = ~np.isnan(train_observed)
        matches = train_observed == observed_labels.reshape(1, -1)
        match_rate = np.divide(
            np.sum(matches & valid, axis=1),
            np.maximum(np.sum(valid, axis=1), 1),
            out=np.zeros(n_users, dtype=np.float32),
            where=np.sum(valid, axis=1) > 0,
        )
        if abs(self.power - 1.0) > 1e-12:
            match_rate = np.power(np.clip(match_rate, 0.0, 1.0), self.power)
        scaled = (match_rate - float(np.max(match_rate))) / max(self.temperature, 1e-6)
        weights = np.exp(scaled).astype(np.float32)
        total = float(np.sum(weights))
        if total <= 1e-8:
            return np.full(n_users, 1.0 / n_users, dtype=np.float32)
        return weights / total

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        weights = self._weights(observed_word_ids, observed_labels)
        prior = self.word_prior[candidate_word_ids]
        if len(weights) == 0:
            return prior.astype(np.float32)
        labels = self.user_word_matrix[:, candidate_word_ids]
        weighted = np.nansum(np.where(np.isnan(labels), 0.0, labels) * weights.reshape(-1, 1), axis=0)
        denom = np.nansum((~np.isnan(labels)).astype(np.float32) * weights.reshape(-1, 1), axis=0)
        vote = np.divide(weighted, denom, out=prior.copy(), where=denom > 1e-8)
        probs = self.prior_blend * prior + (1.0 - self.prior_blend) * vote
        return np.clip(probs.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class CountScaledUserVoteEstimator(Estimator):
    name = "count_scaled_user_vote"

    def __init__(self, count_temperature: float, prior_blend: float, max_logit: float) -> None:
        self.count_temperature = count_temperature
        self.prior_blend = prior_blend
        self.max_logit = max_logit
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

    def _weights(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> np.ndarray:
        n_users = int(self.user_word_matrix.shape[0])
        if n_users == 0:
            return np.zeros(0, dtype=np.float32)
        if len(observed_word_ids) == 0:
            return np.full(n_users, 1.0 / n_users, dtype=np.float32)
        train_observed = self.user_word_matrix[:, observed_word_ids]
        valid = ~np.isnan(train_observed)
        matches = train_observed == observed_labels.reshape(1, -1)
        evidence = np.sum(np.where(matches & valid, 1.0, 0.0), axis=1) - np.sum(np.where((~matches) & valid, 1.0, 0.0), axis=1)
        scaled = evidence / max(self.count_temperature, 1e-6)
        scaled = np.clip(scaled, -self.max_logit, self.max_logit)
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled).astype(np.float32)
        total = float(np.sum(weights))
        if total <= 1e-8:
            return np.full(n_users, 1.0 / n_users, dtype=np.float32)
        return weights / total

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        weights = self._weights(observed_word_ids, observed_labels)
        prior = self.word_prior[candidate_word_ids]
        if len(weights) == 0:
            return prior.astype(np.float32)
        labels = self.user_word_matrix[:, candidate_word_ids]
        weighted = np.nansum(np.where(np.isnan(labels), 0.0, labels) * weights.reshape(-1, 1), axis=0)
        denom = np.nansum((~np.isnan(labels)).astype(np.float32) * weights.reshape(-1, 1), axis=0)
        vote = np.divide(weighted, denom, out=prior.copy(), where=denom > 1e-8)
        probs = self.prior_blend * prior + (1.0 - self.prior_blend) * vote
        return np.clip(probs.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
