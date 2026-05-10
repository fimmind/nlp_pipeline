from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .base import Estimator, UserState


class RaschIRTOnlineEstimator(Estimator):
    name = "rasch_irt_online"

    def __init__(self, prior_var: float, lr: float, n_fit_steps: int) -> None:
        self.prior_var = prior_var
        self.lr = lr
        self.n_fit_steps = n_fit_steps
        self.b = np.array([], dtype=np.float32)
        self.mu = 0.0

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.b = np.zeros(n_words, dtype=np.float32)
        self.mu = 0.0
        if train_responses.empty:
            return
        grp = train_responses.groupby("word_idx")["label"].mean()
        self.b = np.zeros(n_words, dtype=np.float32)
        for idx, p in grp.items():
            p_clip = float(np.clip(p, 1e-4, 1 - 1e-4))
            self.b[int(idx)] = -np.log(p_clip / (1.0 - p_clip))

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(
            payload={
                "theta": 0.0,
                "var": self.prior_var,
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        theta = float(user_state.payload["theta"])
        prev_ids = np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        prev_labels = np.asarray(user_state.payload.get("observed_labels", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        all_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        all_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        for _ in range(self.n_fit_steps):
            logits = theta - self.b[all_ids]
            p = expit(logits)
            grad = np.sum(all_labels - p) - theta / self.prior_var
            h = -np.sum(p * (1 - p)) - 1.0 / self.prior_var
            if abs(h) < 1e-8:
                break
            theta = theta - self.lr * grad / h
        logits = theta - self.b[all_ids]
        p = expit(logits)
        h = -np.sum(p * (1 - p)) - 1.0 / self.prior_var
        var = float(max(1e-6, -1.0 / h))
        return UserState(payload={"theta": theta, "var": var, "observed_word_ids": all_ids, "observed_labels": all_labels})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        return np.clip(expit(user_state.payload["theta"] - self.b[candidate_word_ids]), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p) + user_state.payload["var"] * 0.05


class TwoPLIRTOnlineEstimator(Estimator):
    name = "two_pl_irt_online"

    def __init__(self, prior_var: float, lr: float, n_fit_steps: int) -> None:
        self.prior_var = prior_var
        self.lr = lr
        self.n_fit_steps = n_fit_steps
        self.a = np.array([], dtype=np.float32)
        self.b = np.array([], dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        self.a = np.ones(n_words, dtype=np.float32)
        self.b = np.zeros(n_words, dtype=np.float32)
        if train_responses.empty:
            return
        grp = train_responses.groupby("word_idx")["label"].agg(["mean", "count"])
        for idx, row in grp.iterrows():
            i = int(idx)
            p = float(np.clip(row["mean"], 1e-4, 1 - 1e-4))
            self.b[i] = -np.log(p / (1.0 - p))
            c = float(row["count"])
            self.a[i] = float(np.clip(np.sqrt(c / (c + 20.0)) * 1.7, 0.3, 2.5))

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(
            payload={
                "theta": 0.0,
                "var": self.prior_var,
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        theta = float(user_state.payload["theta"])
        prev_ids = np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        prev_labels = np.asarray(user_state.payload.get("observed_labels", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        all_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        all_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        a = self.a[all_ids]
        b = self.b[all_ids]
        for _ in range(self.n_fit_steps):
            logits = a * (theta - b)
            p = expit(logits)
            grad = float(np.sum(a * (all_labels - p)) - theta / self.prior_var)
            h = float(-np.sum((a ** 2) * p * (1 - p)) - 1.0 / self.prior_var)
            if abs(h) < 1e-8:
                break
            theta = theta - self.lr * grad / h
        logits = a * (theta - b)
        p = expit(logits)
        h = float(-np.sum((a ** 2) * p * (1 - p)) - 1.0 / self.prior_var)
        var = float(max(1e-6, -1.0 / h))
        return UserState(payload={"theta": theta, "var": var, "observed_word_ids": all_ids, "observed_labels": all_labels})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        logits = self.a[candidate_word_ids] * (user_state.payload["theta"] - self.b[candidate_word_ids])
        return np.clip(expit(logits), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p) + user_state.payload["var"] * 0.05
