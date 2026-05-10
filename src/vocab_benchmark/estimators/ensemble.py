from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Estimator, UserState


class WeightedAveragedEnsembleEstimator(Estimator):
    def __init__(self, members: list[Estimator], weights: list[float], name: str, logit_bias: float = 0.0) -> None:
        if len(members) == 0:
            raise ValueError("members must be non-empty")
        if len(members) != len(weights):
            raise ValueError("members and weights must have same length")
        w = np.array(weights, dtype=np.float32)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if float(w.sum()) <= 1e-8:
            raise ValueError("weights sum must be positive")
        self.members = members
        self.weights = (w / w.sum()).astype(np.float32)
        self.name = name
        self.logit_bias = logit_bias

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        for member in self.members:
            member.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        states = [member.initialize_user_state(optional_user_metadata) for member in self.members]
        return UserState(payload={"states": states})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        states = user_state.payload["states"]
        out = []
        for member, state in zip(self.members, states):
            out.append(member.update_user_state(state, observed_word_ids, observed_labels))
        return UserState(payload={"states": out})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        states = user_state.payload["states"]
        probs = []
        for member, state in zip(self.members, states):
            probs.append(member.predict_proba(state, candidate_word_ids))
        stack = np.stack(probs, axis=0)
        p = np.sum(stack * self.weights.reshape(-1, 1), axis=0)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if abs(self.logit_bias) > 1e-12:
            logit = np.log(p) - np.log(1.0 - p)
            p = 1.0 / (1.0 + np.exp(-(logit + self.logit_bias)))
        return np.clip(p, 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)


class BudgetAdaptiveEnsembleEstimator(Estimator):
    def __init__(self, low_budget: Estimator, high_budget: Estimator, switch_observations: int, name: str) -> None:
        if switch_observations <= 0:
            raise ValueError("switch_observations must be positive")
        self.low_budget = low_budget
        self.high_budget = high_budget
        self.switch_observations = switch_observations
        self.name = name

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.low_budget.fit(train_responses, word_features)
        self.high_budget.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(
            payload={
                "low_state": self.low_budget.initialize_user_state(optional_user_metadata),
                "high_state": self.high_budget.initialize_user_state(optional_user_metadata),
                "n_observed": 0,
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        low_state = self.low_budget.update_user_state(user_state.payload["low_state"], observed_word_ids, observed_labels)
        high_state = self.high_budget.update_user_state(user_state.payload["high_state"], observed_word_ids, observed_labels)
        n_observed = int(user_state.payload["n_observed"]) + int(len(observed_word_ids))
        return UserState(payload={"low_state": low_state, "high_state": high_state, "n_observed": n_observed})

    def _active_model_and_state(self, user_state: UserState) -> tuple[Estimator, UserState]:
        if int(user_state.payload["n_observed"]) >= self.switch_observations:
            return self.high_budget, user_state.payload["high_state"]
        return self.low_budget, user_state.payload["low_state"]

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        model, state = self._active_model_and_state(user_state)
        return model.predict_proba(state, candidate_word_ids)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        model, state = self._active_model_and_state(user_state)
        return model.predict_uncertainty(state, candidate_word_ids)
