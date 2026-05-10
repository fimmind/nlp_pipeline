from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from .base import Estimator, UserState
from .svd import SVDRidgeUserEstimator


@dataclass(frozen=True)
class FastTextKernelLogisticConfig:
    embedding_dim: int
    temperature: float
    episodes_per_user: int
    target_samples_per_episode: int
    seed: int
    regularization_c: float
    dynamic_centering_weight: float
    user_rate_centering_weight: float


class FastTextKernelLogisticEstimator(Estimator):
    name = "fasttext_kernel_logistic"

    def __init__(self, config: FastTextKernelLogisticConfig) -> None:
        self.config = config
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.scalar_features = np.empty((0, 0), dtype=np.float32)
        self.word_prior = np.zeros(0, dtype=np.float32)
        self.model: LogisticRegression | None = None

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        if word_features.shape[1] < self.config.embedding_dim + 2:
            raise ValueError("word_features must contain fastText embeddings plus scalar features")
        embeddings = word_features[:, : self.config.embedding_dim].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.maximum(norms, 1e-8)
        scalars = word_features[:, self.config.embedding_dim :].astype(np.float32)
        mean = scalars.mean(axis=0, keepdims=True)
        std = np.maximum(scalars.std(axis=0, keepdims=True), 1e-4)
        self.scalar_features = (scalars - mean) / std
        self._build_word_prior(train_responses)
        if train_responses.empty:
            self.model = None
            return
        x_train, y_train = self._build_training_examples(train_responses)
        if len(y_train) == 0 or len(np.unique(y_train)) < 2:
            self.model = None
            return
        model = LogisticRegression(
            C=self.config.regularization_c,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=self.config.seed,
        )
        model.fit(x_train, y_train)
        self.model = model

    def _build_word_prior(self, train_responses: pd.DataFrame) -> None:
        n_words = self.embeddings.shape[0]
        prior = np.full(n_words, 0.5, dtype=np.float32)
        if not train_responses.empty:
            grouped = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grouped.iterrows():
                prior[int(idx)] = float((row["sum"] + 1.0) / (row["count"] + 2.0))
        self.word_prior = np.clip(prior, 1e-5, 1 - 1e-5)

    def _build_training_examples(self, train_responses: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        grouped = train_responses.sort_values(["user_idx", "word_idx"]).groupby("user_idx")
        for _, group in grouped:
            words = group["word_idx"].to_numpy(dtype=np.int32)
            labels = group["label"].to_numpy(dtype=np.int32)
            if len(words) < 256:
                continue
            for _episode in range(self.config.episodes_per_user):
                order = rng.permutation(len(words))
                shuffled_words = words[order]
                shuffled_labels = labels[order]
                prefix_len = int(rng.choice(np.array([50, 100, 200], dtype=np.int32)))
                if prefix_len >= len(words) - 1:
                    continue
                obs_words = shuffled_words[:prefix_len]
                obs_labels = shuffled_labels[:prefix_len].astype(np.float32)
                target_words = shuffled_words[prefix_len:]
                target_labels = shuffled_labels[prefix_len:]
                target_words, target_labels = self._sample_targets(rng, target_words, target_labels)
                if len(target_words) == 0:
                    continue
                feature_rows.append(self._dynamic_features(obs_words, obs_labels, target_words))
                label_rows.append(target_labels.astype(np.int32))
        if len(feature_rows) == 0:
            return np.empty((0, 8), dtype=np.float32), np.zeros(0, dtype=np.int32)
        return np.concatenate(feature_rows, axis=0), np.concatenate(label_rows, axis=0)

    def _sample_targets(self, rng: np.random.Generator, target_words: np.ndarray, target_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(target_words) <= self.config.target_samples_per_episode:
            return target_words, target_labels
        pos = np.where(target_labels == 1)[0]
        neg = np.where(target_labels == 0)[0]
        half = self.config.target_samples_per_episode // 2
        selected: list[int] = []
        if len(pos) > 0:
            selected.extend(rng.choice(pos, size=min(half, len(pos)), replace=False).tolist())
        if len(neg) > 0:
            selected.extend(rng.choice(neg, size=min(self.config.target_samples_per_episode - len(selected), len(neg)), replace=False).tolist())
        selected_set = set(selected)
        if len(selected) < self.config.target_samples_per_episode:
            remaining = np.array([idx for idx in range(len(target_words)) if idx not in selected_set], dtype=np.int32)
            if len(remaining) > 0:
                selected.extend(rng.choice(remaining, size=min(self.config.target_samples_per_episode - len(selected), len(remaining)), replace=False).tolist())
        selected_idx = np.array(selected, dtype=np.int32)
        return target_words[selected_idx], target_labels[selected_idx]

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _dynamic_features(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray, candidate_word_ids: np.ndarray) -> np.ndarray:
        prior = self.word_prior[candidate_word_ids]
        prior_logit = logit(np.clip(prior, 1e-5, 1 - 1e-5))
        if len(observed_word_ids) == 0:
            kernel = np.zeros(len(candidate_word_ids), dtype=np.float32)
            pos_max = np.zeros(len(candidate_word_ids), dtype=np.float32)
            neg_max = np.zeros(len(candidate_word_ids), dtype=np.float32)
            rate_delta = 0.0
        else:
            obs_emb = self.embeddings[observed_word_ids]
            cand_emb = self.embeddings[candidate_word_ids]
            sim = cand_emb @ obs_emb.T
            scaled = sim / max(self.config.temperature, 1e-6)
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            weights = np.exp(scaled).astype(np.float32)
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
            signed = (2.0 * observed_labels.astype(np.float32) - 1.0).reshape(1, -1)
            kernel = np.sum(weights * signed, axis=1).astype(np.float32)
            pos_mask = observed_labels > 0.5
            neg_mask = observed_labels <= 0.5
            pos_max = sim[:, pos_mask].max(axis=1).astype(np.float32) if np.any(pos_mask) else np.zeros(len(candidate_word_ids), dtype=np.float32)
            neg_max = sim[:, neg_mask].max(axis=1).astype(np.float32) if np.any(neg_mask) else np.zeros(len(candidate_word_ids), dtype=np.float32)
            obs_rate = (float(observed_labels.sum()) + 1.0) / (float(len(observed_labels)) + 2.0)
            expected_rate = float(np.mean(self.word_prior[observed_word_ids]))
            rate_delta = float(logit(np.clip(obs_rate, 1e-5, 1 - 1e-5)) - logit(np.clip(expected_rate, 1e-5, 1 - 1e-5)))
        rate_col = np.full(len(candidate_word_ids), rate_delta, dtype=np.float32)
        scalars = self.scalar_features[candidate_word_ids]
        return np.column_stack([prior_logit, kernel, pos_max, neg_max, pos_max - neg_max, rate_col, scalars]).astype(np.float32)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        obs_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        obs_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        features = self._dynamic_features(obs_ids, obs_labels, candidate_word_ids.astype(np.int32))
        if self.model is None:
            probs = self.word_prior[candidate_word_ids]
        else:
            probs = self.model.predict_proba(features)[:, 1].astype(np.float32)
        logits = logit(np.clip(probs, 1e-6, 1 - 1e-6))
        if abs(self.config.dynamic_centering_weight) > 1e-12:
            logits = logits - self.config.dynamic_centering_weight * float(np.median(logits))
        if abs(self.config.user_rate_centering_weight) > 1e-12 and len(obs_labels) > 0:
            obs_rate = (float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0)
            quantile = float(np.clip(1.0 - obs_rate, 0.05, 0.95))
            logits = logits - self.config.user_rate_centering_weight * float(np.quantile(logits, quantile))
        return np.clip(expit(logits).astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)


@dataclass(frozen=True)
class FastTextSVDRerankerConfig:
    kernel_config: FastTextKernelLogisticConfig
    svd_rank: int
    svd_ridge: float
    svd_residual_scale: float
    svd_intercept_ridge: float
    regularization_c: float
    dynamic_centering_weight: float
    user_rate_centering_weight: float


class FastTextSVDRerankerEstimator(Estimator):
    name = "fasttext_svd_reranker"

    def __init__(self, config: FastTextSVDRerankerConfig) -> None:
        self.config = config
        self.kernel = FastTextKernelLogisticEstimator(config.kernel_config)
        self.svd = SVDRidgeUserEstimator(
            rank=config.svd_rank,
            ridge=config.svd_ridge,
            residual_scale=config.svd_residual_scale,
            intercept_ridge=config.svd_intercept_ridge,
        )
        self.model: LogisticRegression | None = None

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.kernel.fit(train_responses, word_features)
        self.svd.fit(train_responses, word_features)
        x_train, y_train = self._build_training_examples(train_responses)
        if len(y_train) == 0 or len(np.unique(y_train)) < 2:
            self.model = None
            return
        model = LogisticRegression(
            C=self.config.regularization_c,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=self.config.kernel_config.seed,
        )
        model.fit(x_train, y_train)
        self.model = model

    def _build_training_examples(self, train_responses: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.kernel_config.seed + 17)
        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        grouped = train_responses.sort_values(["user_idx", "word_idx"]).groupby("user_idx")
        for _, group in grouped:
            words = group["word_idx"].to_numpy(dtype=np.int32)
            labels = group["label"].to_numpy(dtype=np.int32)
            if len(words) < 256:
                continue
            for _episode in range(self.config.kernel_config.episodes_per_user):
                order = rng.permutation(len(words))
                shuffled_words = words[order]
                shuffled_labels = labels[order]
                prefix_len = int(rng.choice(np.array([50, 100, 200], dtype=np.int32)))
                if prefix_len >= len(words) - 1:
                    continue
                obs_words = shuffled_words[:prefix_len]
                obs_labels = shuffled_labels[:prefix_len].astype(np.float32)
                target_words = shuffled_words[prefix_len:]
                target_labels = shuffled_labels[prefix_len:]
                target_words, target_labels = self.kernel._sample_targets(rng, target_words, target_labels)
                if len(target_words) == 0:
                    continue
                feature_rows.append(self._features(obs_words, obs_labels, target_words))
                label_rows.append(target_labels.astype(np.int32))
        if len(feature_rows) == 0:
            return np.empty((0, 9), dtype=np.float32), np.zeros(0, dtype=np.int32)
        return np.concatenate(feature_rows, axis=0), np.concatenate(label_rows, axis=0)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _features(self, observed_word_ids: np.ndarray, observed_labels: np.ndarray, candidate_word_ids: np.ndarray) -> np.ndarray:
        svd_state = UserState(payload={"observed_word_ids": observed_word_ids.astype(np.int32), "observed_labels": observed_labels.astype(np.float32)})
        svd_probs = self.svd.predict_proba(svd_state, candidate_word_ids.astype(np.int32))
        svd_logits = logit(np.clip(svd_probs, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        kernel_features = self.kernel._dynamic_features(observed_word_ids.astype(np.int32), observed_labels.astype(np.float32), candidate_word_ids.astype(np.int32))
        return np.concatenate([svd_logits.astype(np.float32), kernel_features], axis=1)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        observed_word_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        observed_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        features = self._features(observed_word_ids, observed_labels, candidate_word_ids.astype(np.int32))
        if self.model is None:
            probs = self.svd.predict_proba(
                UserState(payload={"observed_word_ids": observed_word_ids, "observed_labels": observed_labels}),
                candidate_word_ids.astype(np.int32),
            )
        else:
            probs = self.model.predict_proba(features)[:, 1].astype(np.float32)
        logits = logit(np.clip(probs, 1e-6, 1 - 1e-6))
        if abs(self.config.dynamic_centering_weight) > 1e-12:
            logits = logits - self.config.dynamic_centering_weight * float(np.median(logits))
        if abs(self.config.user_rate_centering_weight) > 1e-12 and len(observed_labels) > 0:
            user_rate = (float(observed_labels.sum()) + 1.0) / (float(len(observed_labels)) + 2.0)
            quantile = float(np.clip(1.0 - user_rate, 0.05, 0.95))
            logits = logits - self.config.user_rate_centering_weight * float(np.quantile(logits, quantile))
        return np.clip(expit(logits).astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
