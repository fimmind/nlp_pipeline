from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans

from .estimators.base import Estimator, UserState


class QueryPolicy(ABC):
    name: str

    @abstractmethod
    def select_next_queries(
        self,
        estimator: Estimator,
        user_state: UserState,
        candidate_word_ids: np.ndarray,
        already_queried_word_ids: set[int],
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


class UniformRandomPolicy(QueryPolicy):
    name = "uniform_random"

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = np.array([w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids], dtype=np.int32)
        if len(pool) == 0:
            return np.array([], dtype=np.int32)
        k = min(batch_size, len(pool))
        return rng.choice(pool, size=k, replace=False).astype(np.int32)


class DifficultyStratifiedRandomPolicy(QueryPolicy):
    name = "difficulty_stratified_random"

    def __init__(self, difficulty: np.ndarray, n_bins: int) -> None:
        self.n_bins = n_bins
        qs = np.quantile(difficulty, np.linspace(0, 1, n_bins + 1))
        self.bins = np.clip(np.digitize(difficulty, qs[1:-1], right=True), 0, n_bins - 1)

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        remaining = [w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids]
        if not remaining:
            return np.array([], dtype=np.int32)
        out: list[int] = []
        per_bin = max(1, batch_size // self.n_bins)
        for b in range(self.n_bins):
            bin_words = [w for w in remaining if int(self.bins[w]) == b and w not in out]
            if not bin_words:
                continue
            take = min(per_bin, len(bin_words), batch_size - len(out))
            out.extend(rng.choice(np.array(bin_words), size=take, replace=False).tolist())
            if len(out) >= batch_size:
                break
        if len(out) < batch_size:
            residual = [w for w in remaining if w not in out]
            if residual:
                extra = rng.choice(np.array(residual), size=min(batch_size - len(out), len(residual)), replace=False).tolist()
                out.extend(extra)
        return np.array(out, dtype=np.int32)


class FarthestPointPolicy(QueryPolicy):
    name = "farthest_point"

    def __init__(self, features: np.ndarray) -> None:
        self.features = features

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = [w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids]
        if not pool:
            return np.array([], dtype=np.int32)
        pool_arr = np.array(pool, dtype=np.int32)
        feats = self.features[pool_arr]
        selected_local = [int(rng.integers(0, len(pool_arr)))]
        target = min(batch_size, len(pool_arr))
        d = np.linalg.norm(feats - feats[selected_local[0]], axis=1)
        while len(selected_local) < target:
            next_idx = int(np.argmax(d))
            selected_local.append(next_idx)
            d = np.minimum(d, np.linalg.norm(feats - feats[next_idx], axis=1))
            d[next_idx] = -1.0
        return pool_arr[np.array(selected_local, dtype=np.int32)]


class EntropyPolicy(QueryPolicy):
    name = "entropy"

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = np.array([w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids], dtype=np.int32)
        if len(pool) == 0:
            return np.array([], dtype=np.int32)
        p = estimator.predict_proba(user_state, pool)
        score = -np.abs(p - 0.5)
        idx = np.argsort(-score)[: min(batch_size, len(pool))]
        return pool[idx]


class EmbeddingKMedoidsPolicy(QueryPolicy):
    name = "embedding_kmedoids"

    def __init__(self, features: np.ndarray, n_clusters: int, random_state: int) -> None:
        self.features = features
        n_clusters_eff = max(1, min(n_clusters, len(features)))
        self.kmeans = KMeans(n_clusters=n_clusters_eff, n_init=5, random_state=random_state).fit(features)
        self.medoids = self._compute_medoids()

    def _compute_medoids(self) -> np.ndarray:
        medoids: list[int] = []
        for c in range(self.kmeans.n_clusters):
            members = np.where(self.kmeans.labels_ == c)[0]
            center = self.kmeans.cluster_centers_[c]
            idx = members[np.argmin(np.linalg.norm(self.features[members] - center, axis=1))]
            medoids.append(int(idx))
        return np.array(medoids, dtype=np.int32)

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = set(candidate_word_ids.tolist())
        ranked = [w for w in self.medoids.tolist() if w in pool and w not in already_queried_word_ids]
        if len(ranked) >= batch_size:
            return np.array(ranked[:batch_size], dtype=np.int32)
        remaining = [w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids and w not in ranked]
        if remaining and len(ranked) < batch_size:
            extra = rng.choice(np.array(remaining), size=min(batch_size - len(ranked), len(remaining)), replace=False).tolist()
            ranked.extend(extra)
        return np.array(ranked, dtype=np.int32)
