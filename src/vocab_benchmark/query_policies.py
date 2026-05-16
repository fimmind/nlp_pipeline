from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
from typing import Sequence

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


class UncertaintyPolicy(QueryPolicy):
    name = "uncertainty"

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        pool = np.array([w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids], dtype=np.int32)
        if len(pool) == 0:
            return np.array([], dtype=np.int32)
        uncertainty = estimator.predict_uncertainty(user_state, pool)
        idx = np.argsort(-uncertainty)[: min(batch_size, len(pool))]
        return pool[idx]


@dataclass(frozen=True)
class StochasticTopKEntropyPolicy(QueryPolicy):
    top_k: int
    temperature: float
    name: str = "stochastic_topk_entropy"

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = np.array([w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids], dtype=np.int32)
        if len(pool) == 0:
            return np.array([], dtype=np.int32)
        p = estimator.predict_proba(user_state, pool)
        score = -np.abs(p - 0.5)
        out: list[int] = []
        available = np.ones(len(pool), dtype=bool)
        top_k_eff = max(1, int(self.top_k))
        temp = max(float(self.temperature), 1e-6)
        for _ in range(min(batch_size, len(pool))):
            idx = np.where(available)[0]
            if len(idx) == 0:
                break
            cand = idx[np.argsort(-score[idx])[: min(top_k_eff, len(idx))]]
            logits = score[cand] / temp
            logits = logits - np.max(logits)
            probs = np.exp(np.clip(logits, -60.0, 60.0))
            probs = probs / np.maximum(np.sum(probs), 1e-12)
            choice = int(rng.choice(cand, p=probs))
            out.append(int(pool[choice]))
            available[choice] = False
        return np.array(out, dtype=np.int32)


class ExpectedEntropyReductionPolicy(QueryPolicy):
    name = "expected_entropy_reduction"

    def __init__(self, candidate_pool_size: int, eval_pool_size: int, top_k_stochastic: int, temperature: float) -> None:
        self.candidate_pool_size = candidate_pool_size
        self.eval_pool_size = eval_pool_size
        self.top_k_stochastic = top_k_stochastic
        self.temperature = temperature

    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-8, 1.0 - 1e-8)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    def _clone_state(self, state: UserState) -> UserState:
        return UserState(payload=copy.deepcopy(state.payload))

    def select_next_queries(self, estimator: Estimator, user_state: UserState, candidate_word_ids: np.ndarray, already_queried_word_ids: set[int], batch_size: int, rng: np.random.Generator) -> np.ndarray:
        pool = np.array([w for w in candidate_word_ids.tolist() if w not in already_queried_word_ids], dtype=np.int32)
        if len(pool) == 0:
            return np.array([], dtype=np.int32)
        out: list[int] = []
        temp = max(float(self.temperature), 1e-6)
        top_k_eff = max(1, int(self.top_k_stochastic))
        available = pool.copy()
        for _ in range(min(batch_size, len(pool))):
            if len(available) == 0:
                break
            p_av = estimator.predict_proba(user_state, available)
            unc = self._entropy(p_av)
            pool_k = min(int(self.candidate_pool_size), len(available))
            candidate_local = np.argsort(-unc)[:pool_k]
            candidate_ids = available[candidate_local]
            eval_k = min(int(self.eval_pool_size), len(available))
            eval_local = np.argsort(-unc)[:eval_k]
            eval_ids_full = available[eval_local]
            scores = np.zeros(len(candidate_ids), dtype=np.float64)
            for i, word_id in enumerate(candidate_ids.tolist()):
                p_known = float(np.clip(p_av[candidate_local[i]], 1e-6, 1.0 - 1e-6))
                eval_ids = eval_ids_full[eval_ids_full != int(word_id)]
                if len(eval_ids) == 0:
                    scores[i] = 0.0
                    continue
                s1 = estimator.update_user_state(
                    self._clone_state(user_state),
                    np.array([int(word_id)], dtype=np.int32),
                    np.array([1.0], dtype=np.float32),
                )
                s0 = estimator.update_user_state(
                    self._clone_state(user_state),
                    np.array([int(word_id)], dtype=np.int32),
                    np.array([0.0], dtype=np.float32),
                )
                h1 = float(np.mean(self._entropy(estimator.predict_proba(s1, eval_ids))))
                h0 = float(np.mean(self._entropy(estimator.predict_proba(s0, eval_ids))))
                scores[i] = -(p_known * h1 + (1.0 - p_known) * h0)
            choose_k = min(top_k_eff, len(candidate_ids))
            top_idx = np.argsort(-scores)[:choose_k]
            logits = scores[top_idx] / temp
            logits = logits - np.max(logits)
            probs = np.exp(np.clip(logits, -60.0, 60.0))
            probs = probs / np.maximum(np.sum(probs), 1e-12)
            pick_local = int(rng.choice(top_idx, p=probs))
            chosen = int(candidate_ids[pick_local])
            out.append(chosen)
            available = available[available != chosen]
        return np.array(out, dtype=np.int32)


class StagedAdaptivePolicy(QueryPolicy):
    name = "staged_adaptive"

    def __init__(self, stages: Sequence[tuple[int, QueryPolicy]], fallback_policy: QueryPolicy | None = None) -> None:
        if len(stages) == 0:
            raise ValueError("stages must be non-empty")
        normalized = sorted([(int(max_obs), policy) for max_obs, policy in stages], key=lambda item: item[0])
        self.stages = normalized
        self.fallback_policy = fallback_policy if fallback_policy is not None else normalized[-1][1]

    def _select_policy(self, user_state: UserState) -> QueryPolicy:
        observed_ids = np.asarray(user_state.payload.get("observed_word_ids", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        n_obs = int(len(observed_ids))
        for max_obs, policy in self.stages:
            if n_obs <= max_obs:
                return policy
        return self.fallback_policy

    def select_next_queries(
        self,
        estimator: Estimator,
        user_state: UserState,
        candidate_word_ids: np.ndarray,
        already_queried_word_ids: set[int],
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        policy = self._select_policy(user_state)
        return policy.select_next_queries(
            estimator=estimator,
            user_state=user_state,
            candidate_word_ids=candidate_word_ids,
            already_queried_word_ids=already_queried_word_ids,
            batch_size=batch_size,
            rng=rng,
        )
