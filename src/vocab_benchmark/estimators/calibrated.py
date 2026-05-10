from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score

from .base import Estimator, UserState


class OnlineThresholdWordPriorEstimator(Estimator):
    name = "online_threshold_word_prior"

    def __init__(self, min_observations: int, threshold_blend: float, temperature: float) -> None:
        self.min_observations = min_observations
        self.threshold_blend = threshold_blend
        self.temperature = temperature
        self.word_prior = np.zeros(0, dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = int(word_features.shape[0])
        self.word_prior = np.full(n_words, 0.5, dtype=np.float32)
        if train_responses.empty:
            return
        grouped = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
        word_ids = grouped.index.astype(int).to_numpy()
        prior = (grouped["sum"].to_numpy(dtype=np.float32) + 1.0) / (grouped["count"].to_numpy(dtype=np.float32) + 2.0)
        self.word_prior[word_ids] = np.clip(prior, 1e-5, 1 - 1e-5)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed": {}, "threshold": 0.5})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        observed = dict(user_state.payload["observed"])
        for word_id, label in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            observed[int(word_id)] = int(label)
        threshold = self._fit_threshold(observed)
        return UserState(payload={"observed": observed, "threshold": threshold})

    def _fit_threshold(self, observed: dict[int, int]) -> float:
        if len(observed) < self.min_observations:
            return 0.5
        word_ids = np.array(list(observed.keys()), dtype=np.int32)
        y_true = np.array(list(observed.values()), dtype=np.int32)
        if len(np.unique(y_true)) < 2:
            return 0.5
        scores = self.word_prior[word_ids]
        thresholds = np.unique(scores)
        best_threshold = 0.5
        best_score = -1.0
        for threshold in thresholds.tolist():
            score = float(balanced_accuracy_score(y_true, scores >= threshold))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return float((1.0 - self.threshold_blend) * 0.5 + self.threshold_blend * best_threshold)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        prior = np.clip(self.word_prior[candidate_word_ids], 1e-5, 1 - 1e-5)
        threshold = float(user_state.payload["threshold"])
        score = (logit(prior) - logit(np.clip(threshold, 1e-5, 1 - 1e-5))) / self.temperature
        return np.clip(expit(score).astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class OnlineThresholdCalibratedEstimator(Estimator):
    def __init__(self, base: Estimator, min_observations: int, threshold_blend: float, name: str) -> None:
        self.base = base
        self.min_observations = min_observations
        self.threshold_blend = threshold_blend
        self.name = name

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.base.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(
            payload={
                "base_state": self.base.initialize_user_state(optional_user_metadata),
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
                "threshold": 0.5,
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        base_state = self.base.update_user_state(user_state.payload["base_state"], observed_word_ids, observed_labels)
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        threshold = self._fit_threshold(base_state, new_ids, new_labels)
        return UserState(
            payload={
                "base_state": base_state,
                "observed_word_ids": new_ids,
                "observed_labels": new_labels,
                "threshold": threshold,
            }
        )

    def _fit_threshold(self, base_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> float:
        if len(observed_word_ids) < self.min_observations or len(np.unique(observed_labels)) < 2:
            return 0.5
        scores = self.base.predict_proba(base_state, observed_word_ids)
        thresholds = np.unique(scores)
        best_threshold = 0.5
        best_score = -1.0
        for threshold in thresholds.tolist():
            score = float(balanced_accuracy_score(observed_labels.astype(np.int32), scores >= threshold))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return float((1.0 - self.threshold_blend) * 0.5 + self.threshold_blend * best_threshold)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        base_probs = np.clip(self.base.predict_proba(user_state.payload["base_state"], candidate_word_ids), 1e-6, 1 - 1e-6)
        threshold = float(np.clip(user_state.payload["threshold"], 1e-6, 1 - 1e-6))
        shifted = expit(logit(base_probs) - logit(threshold))
        return np.clip(shifted.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class OnlineAccuracyThresholdCalibratedEstimator(Estimator):
    def __init__(self, base: Estimator, min_observations: int, threshold_blend: float, name: str) -> None:
        self.base = base
        self.min_observations = min_observations
        self.threshold_blend = threshold_blend
        self.name = name

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.base.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(
            payload={
                "base_state": self.base.initialize_user_state(optional_user_metadata),
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
                "threshold": 0.5,
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        base_state = self.base.update_user_state(user_state.payload["base_state"], observed_word_ids, observed_labels)
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        threshold = self._fit_threshold(base_state, new_ids, new_labels)
        return UserState(
            payload={
                "base_state": base_state,
                "observed_word_ids": new_ids,
                "observed_labels": new_labels,
                "threshold": threshold,
            }
        )

    def _fit_threshold(self, base_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> float:
        if len(observed_word_ids) < self.min_observations or len(np.unique(observed_labels)) < 2:
            return 0.5
        scores = self.base.predict_proba(base_state, observed_word_ids)
        thresholds = np.unique(scores)
        best_threshold = 0.5
        best_score = -1.0
        labels = observed_labels.astype(np.int32)
        for threshold in thresholds.tolist():
            score = float(np.mean((scores >= threshold).astype(np.int32) == labels))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return float((1.0 - self.threshold_blend) * 0.5 + self.threshold_blend * best_threshold)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        base_probs = np.clip(self.base.predict_proba(user_state.payload["base_state"], candidate_word_ids), 1e-6, 1 - 1e-6)
        threshold = float(np.clip(user_state.payload["threshold"], 1e-6, 1 - 1e-6))
        shifted = expit(logit(base_probs) - logit(threshold))
        return np.clip(shifted.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class BatchMedianCenteredEstimator(Estimator):
    def __init__(self, base: Estimator, center_weight: float, name: str) -> None:
        self.base = base
        self.center_weight = center_weight
        self.name = name

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.base.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return self.base.initialize_user_state(optional_user_metadata)

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        return self.base.update_user_state(user_state, observed_word_ids, observed_labels)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        probs = np.clip(self.base.predict_proba(user_state, candidate_word_ids), 1e-6, 1 - 1e-6)
        if len(probs) == 0:
            return probs.astype(np.float32)
        logits = logit(probs)
        centered = logits - self.center_weight * float(np.median(logits))
        return np.clip(expit(centered).astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


class ProbabilityPowerCalibratedEstimator(Estimator):
    def __init__(self, base: Estimator, power: float, name: str) -> None:
        self.base = base
        self.power = power
        self.name = name

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.base.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return self.base.initialize_user_state(optional_user_metadata)

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        return self.base.update_user_state(user_state, observed_word_ids, observed_labels)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        probs = np.clip(self.base.predict_proba(user_state, candidate_word_ids), 1e-6, 1 - 1e-6)
        logits = logit(probs) * self.power
        return np.clip(expit(logits).astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
