from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Estimator, UserState


class SVDRidgeUserEstimator(Estimator):
    name = "svd_ridge_user"

    def __init__(self, rank: int, ridge: float, residual_scale: float, intercept_ridge: float) -> None:
        self.rank = rank
        self.ridge = ridge
        self.residual_scale = residual_scale
        self.intercept_ridge = intercept_ridge
        self.word_prior = np.zeros(0, dtype=np.float32)
        self.item_components = np.zeros((0, 0), dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = int(word_features.shape[0])
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        self.item_components = np.zeros((n_words, 0), dtype=np.float32)
        if train_responses.empty:
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

        centered = np.where(np.isnan(matrix), self.word_prior.reshape(1, -1), matrix) - self.word_prior.reshape(1, -1)
        centered = centered.astype(np.float32)
        if centered.shape[0] == 0:
            return
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        effective_rank = min(self.rank, len(singular_values))
        if effective_rank == 0:
            return
        self.item_components = (vt[:effective_rank].T * singular_values[:effective_rank].reshape(1, -1)).astype(np.float32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _solve_user_coefficients(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> tuple[np.ndarray, float]:
        rank = int(self.item_components.shape[1])
        if rank == 0 or len(observed_word_ids) == 0:
            return np.zeros(rank, dtype=np.float32), 0.0
        design = self.item_components[observed_word_ids]
        intercept = np.ones((len(observed_word_ids), 1), dtype=np.float32)
        augmented = np.concatenate([design, intercept], axis=1)
        target = observed_labels.astype(np.float32) - self.word_prior[observed_word_ids]
        penalty = np.eye(rank + 1, dtype=np.float32) * self.ridge
        penalty[-1, -1] = self.intercept_ridge
        lhs = augmented.T @ augmented + penalty
        rhs = augmented.T @ target
        coef = np.linalg.solve(lhs, rhs).astype(np.float32)
        return coef[:rank], float(coef[-1])

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        coef, intercept = self._solve_user_coefficients(observed_word_ids, observed_labels)
        residual = self.item_components[candidate_word_ids] @ coef + intercept if len(coef) > 0 else intercept
        p = self.word_prior[candidate_word_ids] + self.residual_scale * residual
        return np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
