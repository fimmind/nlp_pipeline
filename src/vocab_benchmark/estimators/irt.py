from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


class GroupedResidualIRTOnlineEstimator(Estimator):
    name = "grouped_residual_irt_online"

    def __init__(
        self,
        prior_var: float,
        lr: float,
        n_fit_steps: int,
        n_groups: int,
        grouping_strategy: str,
        group_temperature: float,
        residual_prior_var: float,
        embedding_dim: int,
        random_state: int,
        kmeans_n_init: int,
        pca_components: int,
    ) -> None:
        self.prior_var = prior_var
        self.lr = lr
        self.n_fit_steps = n_fit_steps
        self.n_groups = n_groups
        self.grouping_strategy = grouping_strategy
        self.group_temperature = group_temperature
        self.residual_prior_var = residual_prior_var
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.kmeans_n_init = kmeans_n_init
        self.pca_components = pca_components
        self.b = np.zeros(0, dtype=np.float32)
        self.group_weights = np.zeros((0, 0), dtype=np.float32)
        self.group_prior_mean = np.zeros(0, dtype=np.float32)
        self.group_prior_var = np.zeros(0, dtype=np.float32)
        self._group_cache_signature: tuple[int, int, float, float] | None = None

    def _row_softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted, dtype=np.float64)
        denom = np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)
        return (exp_logits / denom).astype(np.float32)

    def _normalized_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)

    def _build_soft_groups(self, embeddings: np.ndarray) -> np.ndarray:
        if self.grouping_strategy == "kmeans_euclidean":
            model = KMeans(n_clusters=self.n_groups, random_state=self.random_state, n_init=self.kmeans_n_init)
            model.fit(embeddings)
            centers = model.cluster_centers_.astype(np.float32)
            diff = embeddings[:, None, :] - centers[None, :, :]
            dist2 = np.sum(diff * diff, axis=2, dtype=np.float64)
            logits = -dist2 / max(self.group_temperature, 1e-6)
            return self._row_softmax(logits)
        if self.grouping_strategy == "kmeans_cosine":
            norm_embeddings = self._normalized_embeddings(embeddings.astype(np.float32))
            model = KMeans(n_clusters=self.n_groups, random_state=self.random_state, n_init=self.kmeans_n_init)
            model.fit(norm_embeddings)
            centers = model.cluster_centers_.astype(np.float32)
            centers = self._normalized_embeddings(centers)
            similarity = norm_embeddings @ centers.T
            logits = similarity / max(self.group_temperature, 1e-6)
            return self._row_softmax(logits)
        if self.grouping_strategy == "pca_quantile":
            n_components = min(self.pca_components, embeddings.shape[1])
            if n_components < 1:
                raise ValueError("pca_quantile requires at least one embedding component")
            pca = PCA(n_components=n_components, random_state=self.random_state)
            projection = pca.fit_transform(embeddings.astype(np.float64))[:, 0].astype(np.float32)
            quantiles = np.linspace(0.0, 1.0, self.n_groups, endpoint=True)
            centers = np.quantile(projection, quantiles).astype(np.float32)
            if len(centers) > 1:
                spacing = np.diff(np.sort(centers))
                bandwidth = float(np.median(spacing)) if np.any(spacing > 0) else float(np.std(projection))
            else:
                bandwidth = float(np.std(projection))
            bandwidth = max(bandwidth, 1e-3)
            scaled_bandwidth = max(bandwidth * self.group_temperature, 1e-3)
            diff2 = (projection[:, None] - centers[None, :]) ** 2
            logits = -diff2 / (2.0 * scaled_bandwidth * scaled_bandwidth)
            return self._row_softmax(logits)
        if self.grouping_strategy == "anchor_cosine":
            rng = np.random.default_rng(self.random_state)
            norm_embeddings = self._normalized_embeddings(embeddings.astype(np.float32))
            if self.n_groups > norm_embeddings.shape[0]:
                raise ValueError(
                    f"anchor_cosine requires n_groups <= n_words, got n_groups={self.n_groups}, n_words={norm_embeddings.shape[0]}"
                )
            anchor_ids = rng.choice(norm_embeddings.shape[0], size=self.n_groups, replace=False)
            anchors = norm_embeddings[anchor_ids]
            similarity = norm_embeddings @ anchors.T
            logits = similarity / max(self.group_temperature, 1e-6)
            return self._row_softmax(logits)
        raise ValueError(f"unknown grouping_strategy={self.grouping_strategy}")

    def _fit_user_group_residual(self, word_ids: np.ndarray, labels: np.ndarray, prior_prob: np.ndarray) -> np.ndarray:
        if len(word_ids) == 0:
            return np.zeros(self.n_groups, dtype=np.float32)
        x = self.group_weights[word_ids].astype(np.float64)
        y = (labels.astype(np.float64) - prior_prob[word_ids].astype(np.float64)).reshape(-1, 1)
        ridge = np.eye(self.n_groups, dtype=np.float64) * (1.0 / max(self.residual_prior_var, 1e-6))
        lhs = x.T @ x + ridge
        rhs = x.T @ y
        try:
            coef = np.linalg.solve(lhs, rhs).reshape(-1).astype(np.float32)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0].reshape(-1).astype(np.float32)
        return coef

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = word_features.shape[0]
        if word_features.shape[1] < self.embedding_dim:
            raise ValueError(
                f"word_features must include at least embedding_dim={self.embedding_dim} columns, got shape={word_features.shape}"
            )
        embeddings = word_features[:, : self.embedding_dim].astype(np.float32)
        signature = (
            int(embeddings.shape[0]),
            int(embeddings.shape[1]),
            float(embeddings[0, 0]) if embeddings.size > 0 else 0.0,
            float(embeddings[-1, -1]) if embeddings.size > 0 else 0.0,
        )
        self.b = np.zeros(n_words, dtype=np.float32)
        if self._group_cache_signature != signature or self.group_weights.shape != (n_words, self.n_groups):
            self.group_weights = self._build_soft_groups(embeddings)
            self._group_cache_signature = signature
        self.group_prior_mean = np.zeros(self.n_groups, dtype=np.float32)
        self.group_prior_var = np.full(self.n_groups, self.residual_prior_var, dtype=np.float32)
        if train_responses.empty:
            return
        grp = train_responses.groupby("word_idx")["label"].mean()
        for idx, p in grp.items():
            p_clip = float(np.clip(p, 1e-4, 1.0 - 1e-4))
            self.b[int(idx)] = -np.log(p_clip / (1.0 - p_clip))
        prior_prob = expit(-self.b).astype(np.float32)
        user_vectors: list[np.ndarray] = []
        for _, user_rows in train_responses.groupby("user_idx"):
            word_ids = user_rows["word_idx"].to_numpy(dtype=np.int32)
            labels = user_rows["label"].to_numpy(dtype=np.float32)
            if len(word_ids) < max(5, self.n_groups // 2):
                continue
            user_vectors.append(self._fit_user_group_residual(word_ids, labels, prior_prob))
        if len(user_vectors) == 0:
            return
        stacked = np.vstack(user_vectors).astype(np.float32)
        self.group_prior_mean = stacked.mean(axis=0).astype(np.float32)
        var = np.var(stacked, axis=0).astype(np.float32)
        self.group_prior_var = np.clip(var + 1e-3, 1e-3, 25.0).astype(np.float32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        del optional_user_metadata
        return UserState(
            payload={
                "theta": 0.0,
                "var": self.prior_var,
                "group_residual": np.zeros(self.n_groups, dtype=np.float32),
                "group_var_mean": float(np.mean(self.group_prior_var)) if self.group_prior_var.size > 0 else float(self.residual_prior_var),
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
            }
        )

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        prev_labels = np.asarray(user_state.payload.get("observed_labels", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        all_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        all_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        theta = float(user_state.payload.get("theta", 0.0))
        group_residual = np.asarray(
            user_state.payload.get("group_residual", np.zeros(self.n_groups, dtype=np.float32)), dtype=np.float64
        ).reshape(-1)
        if len(group_residual) != self.n_groups:
            group_residual = np.zeros(self.n_groups, dtype=np.float64)
        if len(all_ids) == 0:
            return UserState(
                payload={
                    "theta": theta,
                    "var": float(user_state.payload.get("var", self.prior_var)),
                    "group_residual": group_residual.astype(np.float32),
                    "group_var_mean": float(user_state.payload.get("group_var_mean", np.mean(self.group_prior_var))),
                    "observed_word_ids": all_ids,
                    "observed_labels": all_labels,
                }
            )
        x = self.group_weights[all_ids].astype(np.float64)
        y = all_labels.astype(np.float64)
        prior_mean = self.group_prior_mean.astype(np.float64)
        prior_inv_var = 1.0 / np.maximum(self.group_prior_var.astype(np.float64), 1e-6)
        for _ in range(self.n_fit_steps):
            logits = theta - self.b[all_ids].astype(np.float64) + x @ group_residual
            p = expit(logits)
            w = p * (1.0 - p)
            err = y - p
            grad_theta = float(np.sum(err) - theta / self.prior_var)
            grad_group = x.T @ err - (group_residual - prior_mean) * prior_inv_var
            h_tt = -float(np.sum(w)) - 1.0 / self.prior_var
            h_tg = -(x.T @ w)
            weighted_x = x * w.reshape(-1, 1)
            h_gg = -(x.T @ weighted_x) - np.diag(prior_inv_var)
            h = np.zeros((self.n_groups + 1, self.n_groups + 1), dtype=np.float64)
            h[0, 0] = h_tt
            h[0, 1:] = h_tg
            h[1:, 0] = h_tg
            h[1:, 1:] = h_gg
            grad = np.concatenate([[grad_theta], grad_group], axis=0)
            try:
                delta = np.linalg.solve(h, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(h, grad, rcond=None)[0]
            theta = float(theta - self.lr * delta[0])
            group_residual = group_residual - self.lr * delta[1:]
        logits = theta - self.b[all_ids].astype(np.float64) + x @ group_residual
        p = expit(logits)
        w = p * (1.0 - p)
        h_tt = -float(np.sum(w)) - 1.0 / self.prior_var
        h_tg = -(x.T @ w)
        weighted_x = x * w.reshape(-1, 1)
        h_gg = -(x.T @ weighted_x) - np.diag(prior_inv_var)
        h = np.zeros((self.n_groups + 1, self.n_groups + 1), dtype=np.float64)
        h[0, 0] = h_tt
        h[0, 1:] = h_tg
        h[1:, 0] = h_tg
        h[1:, 1:] = h_gg
        fisher = -h
        try:
            cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fisher)
        theta_var = float(max(1e-6, cov[0, 0]))
        group_var_mean = float(max(1e-6, np.mean(np.diag(cov)[1:])))
        return UserState(
            payload={
                "theta": theta,
                "var": theta_var,
                "group_residual": group_residual.astype(np.float32),
                "group_var_mean": group_var_mean,
                "observed_word_ids": all_ids,
                "observed_labels": all_labels,
            }
        )

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        theta = float(user_state.payload["theta"])
        group_residual = np.asarray(user_state.payload.get("group_residual", np.zeros(self.n_groups, dtype=np.float32)), dtype=np.float64)
        logits = theta - self.b[candidate_word_ids].astype(np.float64) + self.group_weights[candidate_word_ids].astype(np.float64) @ group_residual
        return np.clip(expit(logits), 1e-6, 1.0 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        theta_var = float(user_state.payload.get("var", self.prior_var))
        group_var_mean = float(user_state.payload.get("group_var_mean", self.residual_prior_var))
        return p * (1.0 - p) + theta_var * 0.05 + group_var_mean * 0.02


class BasicRaschFromAccuracyEstimator(Estimator):
    name = "basic_rasch_from_accuracy"

    def __init__(self, prior_var: float, lr: float, n_fit_steps: int, accuracy_values: np.ndarray | None = None) -> None:
        self.prior_var = prior_var
        self.lr = lr
        self.n_fit_steps = n_fit_steps
        self._accuracy_values = accuracy_values
        self.b = np.array([], dtype=np.float32)

    def _load_accuracy_values(self, n_words: int) -> np.ndarray:
        if self._accuracy_values is not None:
            values = np.asarray(self._accuracy_values, dtype=np.float32).reshape(-1)
            if len(values) != n_words:
                raise ValueError(f"accuracy_values length mismatch: expected {n_words}, got {len(values)}")
            return values
        freq = pd.read_csv("data/processed/frequency.csv")
        if "accuracy" not in freq.columns:
            raise ValueError("required column missing: data/processed/frequency.csv::accuracy")
        values = pd.to_numeric(freq["accuracy"], errors="coerce").to_numpy(dtype=np.float32)
        if len(values) < n_words:
            raise ValueError(
                f"insufficient accuracy rows in data/processed/frequency.csv: need {n_words}, got {len(values)}"
            )
        return values[:n_words]

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        del train_responses
        n_words = word_features.shape[0]
        raw_accuracy = self._load_accuracy_values(n_words)
        accuracy = np.where(raw_accuracy > 1.0, raw_accuracy / 100.0, raw_accuracy)
        accuracy = np.where(np.isnan(accuracy), 0.5, accuracy).astype(np.float32)
        p = np.clip(accuracy, 1e-4, 1.0 - 1e-4)
        self.b = -np.log(p / (1.0 - p)).astype(np.float32)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        del optional_user_metadata
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
            h = -np.sum(p * (1.0 - p)) - 1.0 / self.prior_var
            if abs(h) < 1e-8:
                break
            theta = theta - self.lr * grad / h
        logits = theta - self.b[all_ids]
        p = expit(logits)
        h = -np.sum(p * (1.0 - p)) - 1.0 / self.prior_var
        var = float(max(1e-6, -1.0 / h))
        return UserState(payload={"theta": theta, "var": var, "observed_word_ids": all_ids, "observed_labels": all_labels})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        return np.clip(expit(user_state.payload["theta"] - self.b[candidate_word_ids]), 1e-6, 1.0 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p) + user_state.payload["var"] * 0.05
