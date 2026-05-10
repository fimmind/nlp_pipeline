from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .base import Estimator, UserState


class LowRankMFOnlineEstimator(Estimator):
    name = "low_rank_mf_online"

    def __init__(self, rank: int, n_epochs: int, lr: float, reg: float, seed: int) -> None:
        self.rank = rank
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.beta = np.array([], dtype=np.float32)
        self.v = np.array([], dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        rng = np.random.default_rng(self.seed)
        self.beta = np.zeros(n_words, dtype=np.float32)
        self.v = (0.05 * rng.normal(size=(n_words, self.rank))).astype(np.float32)
        if train_responses.empty:
            return
        users = sorted(train_responses["user_idx"].unique().tolist())
        user_to_row = {int(u): i for i, u in enumerate(users)}
        z = np.zeros((len(users), self.rank), dtype=np.float32)
        alpha = np.zeros(len(users), dtype=np.float32)
        rows = train_responses[["user_idx", "word_idx", "label"]].to_numpy()
        for _ in range(self.n_epochs):
            rng.shuffle(rows)
            for u_raw, w_raw, y_raw in rows:
                u = user_to_row[int(u_raw)]
                w = int(w_raw)
                y = float(y_raw)
                logit = alpha[u] + self.beta[w] + float(np.dot(z[u], self.v[w]))
                p = float(expit(logit))
                e = y - p
                z_u = z[u].copy()
                v_w = self.v[w].copy()
                alpha[u] += self.lr * (e - self.reg * alpha[u])
                self.beta[w] += self.lr * (e - self.reg * self.beta[w])
                z[u] += self.lr * (e * v_w - self.reg * z_u)
                self.v[w] += self.lr * (e * z_u - self.reg * v_w)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"alpha": 0.0, "z": np.zeros(self.rank, dtype=np.float32), "n": 0})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        alpha = float(user_state.payload["alpha"])
        z = user_state.payload["z"].copy()
        n = int(user_state.payload["n"])
        for w, y in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            w_idx = int(w)
            y_val = float(y)
            logit = alpha + float(self.beta[w_idx]) + float(np.dot(z, self.v[w_idx]))
            p = float(expit(logit))
            e = y_val - p
            z_old = z.copy()
            alpha += self.lr * (e - self.reg * alpha)
            z += self.lr * (e * self.v[w_idx] - self.reg * z_old)
            n += 1
        return UserState(payload={"alpha": alpha, "z": z, "n": n})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        alpha = float(user_state.payload["alpha"])
        z = user_state.payload["z"]
        logits = alpha + self.beta[candidate_word_ids] + self.v[candidate_word_ids] @ z
        return np.clip(expit(logits), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        n = max(1, int(user_state.payload["n"]))
        return p * (1 - p) + 1.0 / (n + 1.0)
