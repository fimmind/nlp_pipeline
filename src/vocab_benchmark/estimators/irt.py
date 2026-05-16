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
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=self.n_groups, random_state=self.random_state, n_init=self.kmeans_n_init)
            model.fit(embeddings)
            centers = model.cluster_centers_.astype(np.float32)
            diff = embeddings[:, None, :] - centers[None, :, :]
            dist2 = np.sum(diff * diff, axis=2, dtype=np.float64)
            logits = -dist2 / max(self.group_temperature, 1e-6)
            return self._row_softmax(logits)
        if self.grouping_strategy == "kmeans_cosine":
            from sklearn.cluster import KMeans

            norm_embeddings = self._normalized_embeddings(embeddings.astype(np.float32))
            model = KMeans(n_clusters=self.n_groups, random_state=self.random_state, n_init=self.kmeans_n_init)
            model.fit(norm_embeddings)
            centers = model.cluster_centers_.astype(np.float32)
            centers = self._normalized_embeddings(centers)
            similarity = norm_embeddings @ centers.T
            logits = similarity / max(self.group_temperature, 1e-6)
            return self._row_softmax(logits)
        if self.grouping_strategy == "pca_quantile":
            from sklearn.decomposition import PCA

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


class Response12GroupedResidualIRTEstimator(Estimator):
    name = "response12_grouped_residual_irt"

    def __init__(
        self,
        tau_theta: float,
        tau_delta: float,
        gate_c: float,
        n_groups: int,
        random_state: int,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        threshold_shrink_c: float,
        accuracy_values: np.ndarray | None = None,
        use_accuracy_difficulty: bool = True,
    ) -> None:
        self.tau_theta = tau_theta
        self.tau_delta = tau_delta
        self.gate_c = gate_c
        self.n_groups = n_groups
        self.random_state = random_state
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.threshold_shrink_c = threshold_shrink_c
        self.accuracy_values = accuracy_values
        self.use_accuracy_difficulty = use_accuracy_difficulty
        self.b = np.zeros(0, dtype=np.float32)
        self.q_matrix = np.zeros((0, 0), dtype=np.float32)

    def _load_accuracy_values(self, n_words: int) -> np.ndarray:
        if self.accuracy_values is not None:
            values = np.asarray(self.accuracy_values, dtype=np.float64).reshape(-1)
            if len(values) != n_words:
                raise ValueError(f"accuracy_values length mismatch: expected {n_words}, got {len(values)}")
            return values
        freq = pd.read_csv("data/processed/frequency.csv")
        if "accuracy" not in freq.columns:
            raise ValueError("required column missing: data/processed/frequency.csv::accuracy")
        values = pd.to_numeric(freq["accuracy"], errors="coerce").to_numpy(dtype=np.float64)
        if len(values) < n_words:
            raise ValueError(
                f"insufficient accuracy rows in data/processed/frequency.csv: need {n_words}, got {len(values)}"
            )
        return values[:n_words]

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        n_words = int(word_features.shape[0])
        self.b = np.zeros(n_words, dtype=np.float32)
        self.q_matrix = np.zeros((n_words, self.n_groups), dtype=np.float32)
        if train_responses.empty:
            if n_words > 0:
                self.q_matrix[:] = 1.0 / float(self.n_groups)
            return

        grouped = train_responses.groupby("word_idx")["label"].mean()
        observed_word_ids = np.array(sorted(grouped.index.astype(int).tolist()), dtype=np.int32)
        if self.use_accuracy_difficulty:
            raw_accuracy = self._load_accuracy_values(n_words)
            accuracy = np.where(raw_accuracy > 1.0, raw_accuracy / 100.0, raw_accuracy)
            accuracy = np.where(np.isnan(accuracy), 0.5, accuracy)
            prior_p = np.clip(accuracy, 1e-4, 1.0 - 1e-4).astype(np.float64)
        else:
            prior_p = np.full(n_words, 0.5, dtype=np.float64)
            for idx, p in grouped.items():
                prior_p[int(idx)] = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        b_raw = -np.log(prior_p / (1.0 - prior_p))
        if len(observed_word_ids) > 0:
            mean_b = float(np.mean(b_raw[observed_word_ids]))
            std_b = float(np.std(b_raw[observed_word_ids]))
        else:
            mean_b = float(np.mean(b_raw))
            std_b = float(np.std(b_raw))
        if std_b <= 1e-8:
            self.b = np.zeros(n_words, dtype=np.float32)
        else:
            self.b = ((b_raw - mean_b) / std_b).astype(np.float32)

        users = sorted(train_responses["user_idx"].astype(int).unique().tolist())
        user_to_row = {user_id: row_idx for row_idx, user_id in enumerate(users)}
        y = np.full((len(users), n_words), np.nan, dtype=np.float32)
        for row in train_responses[["user_idx", "word_idx", "label"]].itertuples(index=False):
            y[user_to_row[int(row.user_idx)], int(row.word_idx)] = float(row.label)
        # Dense binary matrix by item-majority imputation.
        counts = np.sum(~np.isnan(y), axis=0)
        sums = np.nansum(y, axis=0)
        item_mean = np.divide(sums, np.maximum(counts, 1), out=np.full(n_words, 0.5, dtype=np.float32), where=counts > 0)
        item_fill = (item_mean >= 0.5).astype(np.float32)
        y_dense = np.where(np.isnan(y), item_fill.reshape(1, -1), y).astype(np.float32)
        if len(observed_word_ids) == 0:
            self.q_matrix[:] = 1.0 / float(self.n_groups)
            return
        q_observed = self._build_response12_groups(y_dense[:, observed_word_ids]).astype(np.float32)
        self.q_matrix[:] = 1.0 / float(self.n_groups)
        self.q_matrix[observed_word_ids] = q_observed

    def _build_response12_groups(self, y_dense: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans

        x = y_dense.T.astype(np.float32)  # items x users
        row_mean = x.mean(axis=1, keepdims=True)
        x_centered = x - row_mean
        norms = np.linalg.norm(x_centered, axis=1, keepdims=True)
        x_norm = x_centered / np.maximum(norms, 1e-8)
        model = KMeans(n_clusters=self.n_groups, random_state=self.random_state, n_init=20)
        model.fit(x_norm)
        centers = model.cluster_centers_.astype(np.float32)
        center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(center_norms, 1e-8)
        sim = x_norm @ centers.T
        q = np.zeros_like(sim, dtype=np.float32)
        for i in range(sim.shape[0]):
            row = sim[i]
            top = np.argpartition(-row, kth=min(2, len(row) - 1))[:3]
            top = top[np.argsort(-row[top])]
            logits = row[top] * 6.0
            logits = logits - np.max(logits)
            probs = np.exp(logits, dtype=np.float64)
            probs = probs / np.maximum(np.sum(probs), 1e-12)
            probs = probs.astype(np.float32)
            hard_idx = int(top[0])
            hard_pos = int(np.where(top == hard_idx)[0][0])
            probs[hard_pos] = max(float(probs[hard_pos]), 0.5)
            probs = probs / np.maximum(np.sum(probs), 1e-12)
            q[i, top] = probs
        row_sum = q.sum(axis=1, keepdims=True)
        return q / np.maximum(row_sum, 1e-8)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        del optional_user_metadata
        return UserState(
            payload={
                "theta": 0.0,
                "delta": np.zeros(self.n_groups, dtype=np.float32),
                "threshold": 0.5,
                "var": float(self.tau_theta**2),
                "observed_word_ids": np.zeros(0, dtype=np.int32),
                "observed_labels": np.zeros(0, dtype=np.float32),
            }
        )

    def _fit_theta_init(self, word_ids: np.ndarray, labels: np.ndarray) -> float:
        from scipy.optimize import minimize_scalar

        if len(word_ids) == 0:
            return 0.0
        b_obs = self.b[word_ids].astype(np.float64)
        y_obs = labels.astype(np.float64)

        def objective(theta: float) -> float:
            logits = theta - b_obs
            nll = np.sum(y_obs * np.logaddexp(0.0, -logits) + (1.0 - y_obs) * np.logaddexp(0.0, logits))
            prior = theta * theta / (2.0 * self.tau_theta * self.tau_theta)
            return float(nll + prior)

        result = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-5})
        if not result.success:
            raise RuntimeError(f"Rasch theta optimization failed: {result.message}")
        return float(result.x)

    def _class_weights(self, labels: np.ndarray) -> np.ndarray:
        pos_rate = float(np.clip(np.mean(labels), 0.05, 0.95))
        w_pos = 0.5 / pos_rate
        w_neg = 0.5 / (1.0 - pos_rate)
        return (labels * w_pos + (1.0 - labels) * w_neg).astype(np.float64)

    def _objective_and_grad(
        self, params: np.ndarray, word_ids: np.ndarray, labels: np.ndarray, gate: float, sample_weights: np.ndarray
    ) -> tuple[float, np.ndarray]:
        theta = float(params[0])
        delta = params[1:]
        b_obs = self.b[word_ids].astype(np.float64)
        q_obs = self.q_matrix[word_ids].astype(np.float64)
        residual = q_obs @ delta
        z = theta - b_obs + gate * residual
        p = expit(z)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        nll = -np.sum(sample_weights * (labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))
        prior = theta * theta / (2.0 * self.tau_theta * self.tau_theta) + float(
            np.sum(delta * delta) / (2.0 * self.tau_delta * self.tau_delta)
        )
        err = sample_weights * (p - labels)
        grad_theta = float(np.sum(err) + theta / (self.tau_theta * self.tau_theta))
        grad_delta = gate * (q_obs.T @ err) + delta / (self.tau_delta * self.tau_delta)
        grad = np.concatenate([[grad_theta], grad_delta], axis=0).astype(np.float64)
        return float(nll + prior), grad

    def _balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pos = y_true == 1
        neg = y_true == 0
        if int(pos.sum()) == 0 or int(neg.sum()) == 0:
            return 0.5
        tpr = float(np.mean(y_pred[pos] == 1))
        tnr = float(np.mean(y_pred[neg] == 0))
        return 0.5 * (tpr + tnr)

    def _optimized_threshold(self, probs: np.ndarray, labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.5
        thresholds = np.arange(self.threshold_min, self.threshold_max + 0.5 * self.threshold_step, self.threshold_step)
        best_t = 0.5
        best_ba = -1.0
        for threshold in thresholds.tolist():
            pred = (probs >= float(threshold)).astype(np.int32)
            ba = self._balanced_accuracy(labels, pred)
            if ba > best_ba + 1e-12:
                best_ba = ba
                best_t = float(threshold)
            elif abs(ba - best_ba) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_t - 0.5):
                best_t = float(threshold)
        shrink = float(len(labels)) / float(len(labels) + self.threshold_shrink_c)
        return float(0.5 + shrink * (best_t - 0.5))

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        from scipy.optimize import minimize

        prev_ids = np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        prev_labels = np.asarray(user_state.payload.get("observed_labels", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        all_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        all_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        if len(all_ids) == 0:
            return user_state

        gate = float(len(all_ids)) / float(len(all_ids) + self.gate_c)
        sample_weights = self._class_weights(all_labels)
        theta_init = self._fit_theta_init(all_ids, all_labels)
        delta_init = np.asarray(user_state.payload.get("delta", np.zeros(self.n_groups, dtype=np.float32)), dtype=np.float64)
        if delta_init.shape[0] != self.n_groups:
            delta_init = np.zeros(self.n_groups, dtype=np.float64)
        init = np.concatenate([[theta_init], delta_init], axis=0).astype(np.float64)

        def objective(params: np.ndarray) -> float:
            value, _ = self._objective_and_grad(params=params, word_ids=all_ids, labels=all_labels, gate=gate, sample_weights=sample_weights)
            return value

        def gradient(params: np.ndarray) -> np.ndarray:
            _, grad = self._objective_and_grad(params=params, word_ids=all_ids, labels=all_labels, gate=gate, sample_weights=sample_weights)
            return grad

        result = minimize(
            fun=objective,
            x0=init,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-9},
        )
        params = result.x if result.success else init
        theta = float(params[0])
        delta = params[1:].astype(np.float32)
        logits_obs = theta - self.b[all_ids].astype(np.float64) + gate * (self.q_matrix[all_ids].astype(np.float64) @ delta.astype(np.float64))
        probs_obs = np.clip(expit(logits_obs).astype(np.float32), 1e-6, 1.0 - 1e-6)
        threshold = self._optimized_threshold(probs=probs_obs, labels=all_labels.astype(np.int32))
        var = float(max(1e-6, self.tau_theta * self.tau_theta / (1.0 + len(all_ids))))
        return UserState(
            payload={
                "theta": theta,
                "delta": delta,
                "threshold": threshold,
                "var": var,
                "observed_word_ids": all_ids,
                "observed_labels": all_labels,
            }
        )

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        theta = float(user_state.payload["theta"])
        delta = np.asarray(user_state.payload["delta"], dtype=np.float64)
        obs_count = int(len(np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)))))
        gate = float(obs_count) / float(obs_count + self.gate_c)
        logits = theta - self.b[candidate_word_ids].astype(np.float64) + gate * (self.q_matrix[candidate_word_ids].astype(np.float64) @ delta)
        return np.clip(expit(logits).astype(np.float32), 1e-6, 1.0 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        var = float(user_state.payload.get("var", self.tau_theta * self.tau_theta))
        return p * (1.0 - p) + 0.05 * var
