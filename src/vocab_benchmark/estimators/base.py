from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UserState:
    payload: dict[str, Any]


class Estimator(ABC):
    name: str

    @abstractmethod
    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize_user_state(self, optional_user_metadata: dict[str, Any] | None = None) -> UserState:
        raise NotImplementedError

    @abstractmethod
    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError
