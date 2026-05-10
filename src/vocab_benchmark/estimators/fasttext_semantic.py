from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from .base import Estimator, UserState


@dataclass(frozen=True)
class FastTextSemanticConfig:
    embedding_dim: int
    projection_dim: int
    scalar_dim: int
    hidden_dim: int
    lr: float
    dropout: float
    max_epochs: int
    seed: int
    weight_decay: float
    target_batch_size: int
    min_prefix_len: int
    max_prefix_len: int
    prior_logit_weight: float
    user_rate_weight: float
    dynamic_centering_weight: float
    user_rate_centering_weight: float
    memory_weight: float
    prototype_weight: float
    hard_negative_weight: float
    hard_negative_k: int
    early_stopping_patience: int


class _FastTextSemanticModule(nn.Module):
    def __init__(self, config: FastTextSemanticConfig, scalar_input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.embedding_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_input_dim, config.scalar_dim),
            nn.GELU(),
            nn.LayerNorm(config.scalar_dim),
        )
        item_dim = config.projection_dim + config.scalar_dim
        self.item_bias = nn.Sequential(
            nn.Linear(item_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.prototype_transform = nn.Linear(config.projection_dim, config.projection_dim, bias=False)
        self.key_proj = nn.Linear(config.projection_dim + 1, config.projection_dim)
        self.query_proj = nn.Linear(config.projection_dim, config.projection_dim)
        self.memory_scale = nn.Parameter(torch.tensor(config.memory_weight, dtype=torch.float32))
        self.prototype_scale = nn.Parameter(torch.tensor(config.prototype_weight, dtype=torch.float32))

    def encode_items(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = features[:, : self.config.embedding_dim]
        scalars = features[:, self.config.embedding_dim :]
        z = self.embedding_proj(embeddings)
        z = f.normalize(z, p=2.0, dim=-1)
        s = self.scalar_proj(scalars)
        item = torch.cat([z, s], dim=-1)
        return z, s, item

    def _prototype_delta(self, obs_z: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        if obs_z.shape[0] == 0:
            return torch.zeros(self.config.projection_dim, device=obs_z.device)
        pos_mask = obs_labels > 0.5
        neg_mask = obs_labels <= 0.5
        pos = obs_z[pos_mask].mean(dim=0) if int(pos_mask.sum().item()) > 0 else torch.zeros(self.config.projection_dim, device=obs_z.device)
        neg = obs_z[neg_mask].mean(dim=0) if int(neg_mask.sum().item()) > 0 else torch.zeros(self.config.projection_dim, device=obs_z.device)
        return pos - neg

    def _memory_score(self, target_z: torch.Tensor, obs_z: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        if obs_z.shape[0] == 0:
            return torch.zeros(target_z.shape[0], device=target_z.device)
        signed = (2.0 * obs_labels - 1.0).view(-1, 1)
        keys = torch.tanh(self.key_proj(torch.cat([obs_z, signed], dim=-1)))
        queries = torch.tanh(self.query_proj(target_z))
        scale = float(np.sqrt(max(1, self.config.projection_dim)))
        attn = torch.softmax((queries @ keys.transpose(0, 1)) / scale, dim=-1)
        return (attn @ signed).squeeze(-1)

    def score(self, target_features: torch.Tensor, obs_features: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        target_z, _, target_item = self.encode_items(target_features)
        obs_z = self.encode_items(obs_features)[0] if obs_features.shape[0] > 0 else torch.zeros((0, self.config.projection_dim), device=target_features.device)
        delta = self._prototype_delta(obs_z, obs_labels)
        prototype_direction = self.prototype_transform(delta.view(1, -1)).view(-1)
        prototype_score = target_z @ prototype_direction
        memory_score = self._memory_score(target_z, obs_z, obs_labels)
        bias = self.item_bias(target_item).squeeze(-1)
        return bias + self.prototype_scale * prototype_score + self.memory_scale * memory_score


class FastTextSemanticPrototypeEstimator(Estimator):
    name = "fasttext_semantic_prototype"

    def __init__(self, config: FastTextSemanticConfig) -> None:
        self.config = config
        self.model: _FastTextSemanticModule | None = None
        self.word_features = np.zeros((0, 0), dtype=np.float32)
        self.scaler_mean = np.zeros(2, dtype=np.float32)
        self.scaler_std = np.ones(2, dtype=np.float32)
        self.word_prior_logit = np.zeros(0, dtype=np.float32)
        self.neighbors = np.empty((0, 0), dtype=np.int32)
        self.decision_logit_bias = 0.0

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        if word_features.shape[1] < self.config.embedding_dim + 2:
            raise ValueError("word_features must contain fastText embeddings plus two scalar features")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.use_deterministic_algorithms(True)
        self._set_features(word_features)
        self._build_word_prior_logit(train_responses)
        self._build_neighbors()
        scalar_input_dim = int(self.word_features.shape[1] - self.config.embedding_dim)
        self.model = _FastTextSemanticModule(self.config, scalar_input_dim)
        if train_responses.empty:
            return
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(self.config.seed)
        user_sequences = self._build_user_sequences(train_responses)
        best_loss = float("inf")
        stale = 0
        for _epoch in range(self.config.max_epochs):
            rng.shuffle(user_sequences)
            losses: list[float] = []
            for words, labels in user_sequences:
                if len(words) <= self.config.min_prefix_len + 1:
                    continue
                order = rng.permutation(len(words))
                shuffled_words = words[order]
                shuffled_labels = labels[order]
                prefix_len = int(rng.integers(self.config.min_prefix_len, min(len(words) - 1, self.config.max_prefix_len) + 1))
                obs_words = shuffled_words[:prefix_len]
                obs_labels = shuffled_labels[:prefix_len]
                target_words = shuffled_words[prefix_len:]
                target_labels = shuffled_labels[prefix_len:]
                target_words, target_labels = self._sample_targets(rng, target_words, target_labels)
                if len(target_words) == 0:
                    continue
                obs_feat = torch.from_numpy(self.word_features[obs_words])
                obs_lab = torch.from_numpy(obs_labels.astype(np.float32))
                target_feat = torch.from_numpy(self.word_features[target_words])
                y_true = torch.from_numpy(target_labels.astype(np.float32))
                logits = self._score_with_context(target_feat, obs_feat, obs_lab, target_words, obs_labels)
                pos_count = float(np.sum(target_labels == 1))
                neg_count = float(np.sum(target_labels == 0))
                pos_weight = torch.tensor([(neg_count + 1.0) / (pos_count + 1.0)], dtype=torch.float32)
                bce = f.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pos_weight)
                p = torch.sigmoid(logits)
                pos_mask = y_true > 0.5
                neg_mask = y_true <= 0.5
                ba_loss = torch.tensor(0.0)
                if int(pos_mask.sum().item()) > 0 and int(neg_mask.sum().item()) > 0:
                    ba_loss = 0.5 * ((1.0 - p[pos_mask]).mean() + p[neg_mask].mean())
                hard_negative = self._hard_negative_loss(target_words, target_labels, logits, obs_feat, obs_lab, obs_labels)
                loss = bce + 0.25 * ba_loss + self.config.hard_negative_weight * hard_negative
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if len(losses) == 0:
                break
            mean_loss = float(np.mean(losses))
            if mean_loss + 1e-6 < best_loss:
                best_loss = mean_loss
                stale = 0
            else:
                stale += 1
                if stale >= self.config.early_stopping_patience:
                    break
        self.decision_logit_bias = self._fit_decision_logit_bias(train_responses, rng)

    def _set_features(self, word_features: np.ndarray) -> None:
        features = word_features.astype(np.float32).copy()
        scalars = features[:, self.config.embedding_dim :]
        self.scaler_mean = scalars.mean(axis=0).astype(np.float32)
        self.scaler_std = np.maximum(scalars.std(axis=0), 1e-4).astype(np.float32)
        features[:, self.config.embedding_dim :] = (scalars - self.scaler_mean.reshape(1, -1)) / self.scaler_std.reshape(1, -1)
        self.word_features = features

    def _build_word_prior_logit(self, train_responses: pd.DataFrame) -> None:
        n_words = self.word_features.shape[0]
        prior = np.full(n_words, 0.5, dtype=np.float32)
        if not train_responses.empty:
            grouped = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grouped.iterrows():
                prior[int(idx)] = float((row["sum"] + 1.0) / (row["count"] + 2.0))
        prior = np.clip(prior, 1e-5, 1 - 1e-5)
        self.word_prior_logit = (np.log(prior) - np.log(1.0 - prior)).astype(np.float32)

    def _build_neighbors(self) -> None:
        n_words = self.word_features.shape[0]
        if n_words == 0:
            self.neighbors = np.empty((0, 0), dtype=np.int32)
            return
        embeddings = self.word_features[:, : self.config.embedding_dim]
        k = min(max(1, self.config.hard_negative_k * 6), n_words - 1 if n_words > 1 else 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn_model.fit(embeddings)
        self.neighbors = nn_model.kneighbors(return_distance=False)[:, 1:].astype(np.int32)

    def _build_user_sequences(self, train_responses: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        sequences: list[tuple[np.ndarray, np.ndarray]] = []
        grouped = train_responses.sort_values(["user_idx", "word_idx"]).groupby("user_idx")
        for _, group in grouped:
            sequences.append((group["word_idx"].to_numpy(dtype=np.int32), group["label"].to_numpy(dtype=np.int32)))
        return sequences

    def _sample_targets(self, rng: np.random.Generator, target_words: np.ndarray, target_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(target_words) <= self.config.target_batch_size:
            return target_words, target_labels
        pos = np.where(target_labels == 1)[0]
        neg = np.where(target_labels == 0)[0]
        half = self.config.target_batch_size // 2
        selected: list[int] = []
        if len(pos) > 0:
            selected.extend(rng.choice(pos, size=min(half, len(pos)), replace=False).tolist())
        if len(neg) > 0:
            selected.extend(rng.choice(neg, size=min(self.config.target_batch_size - len(selected), len(neg)), replace=False).tolist())
        if len(selected) < self.config.target_batch_size:
            remaining = np.array([i for i in range(len(target_words)) if i not in set(selected)], dtype=np.int32)
            if len(remaining) > 0:
                selected.extend(rng.choice(remaining, size=min(self.config.target_batch_size - len(selected), len(remaining)), replace=False).tolist())
        selected_arr = np.array(selected, dtype=np.int32)
        return target_words[selected_arr], target_labels[selected_arr]

    def _score_with_context(
        self,
        target_feat: torch.Tensor,
        obs_feat: torch.Tensor,
        obs_lab: torch.Tensor,
        target_words: np.ndarray,
        obs_labels: np.ndarray,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        logits = self.model.score(target_feat, obs_feat, obs_lab)
        prior_logits = torch.from_numpy(self.word_prior_logit[target_words])
        user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0)) if len(obs_labels) > 0 else 0.5
        user_bias = self.config.user_rate_weight * (user_rate - 0.5)
        return logits + self.config.prior_logit_weight * prior_logits + user_bias

    def _hard_negative_loss(
        self,
        target_words: np.ndarray,
        target_labels: np.ndarray,
        logits: torch.Tensor,
        obs_feat: torch.Tensor,
        obs_lab: torch.Tensor,
        obs_labels: np.ndarray,
    ) -> torch.Tensor:
        if self.model is None or self.config.hard_negative_weight <= 0.0:
            return torch.tensor(0.0)
        pos_positions = np.where(target_labels == 1)[0]
        target_map = {int(word): int(label) for word, label in zip(target_words.tolist(), target_labels.tolist())}
        losses: list[torch.Tensor] = []
        for pos_pos in pos_positions.tolist()[:16]:
            pos_word = int(target_words[pos_pos])
            neg_words = [int(word) for word in self.neighbors[pos_word].tolist() if target_map.get(int(word), -1) == 0]
            if len(neg_words) == 0:
                continue
            neg_sel = np.array(neg_words[: self.config.hard_negative_k], dtype=np.int32)
            neg_feat = torch.from_numpy(self.word_features[neg_sel])
            neg_logits = self._score_with_context(neg_feat, obs_feat, obs_lab, neg_sel, obs_labels)
            losses.append(torch.relu(torch.tensor(1.0) - (logits[pos_pos] - torch.max(neg_logits))))
        if len(losses) == 0:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def _fit_decision_logit_bias(self, train_responses: pd.DataFrame, rng: np.random.Generator) -> float:
        users = sorted(train_responses["user_idx"].astype(int).unique().tolist())
        probs_all: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        for user_idx in users[: min(8, len(users))]:
            rows = train_responses[train_responses["user_idx"] == user_idx].sort_values("word_idx")
            words = rows["word_idx"].to_numpy(dtype=np.int32)
            labels = rows["label"].to_numpy(dtype=np.int32)
            if len(words) < 256:
                continue
            order = rng.permutation(len(words))
            words = words[order]
            labels = labels[order]
            prefix_len = min(200, max(50, len(words) // 10))
            rem_ids = words[prefix_len:]
            rem_labels = labels[prefix_len:]
            if len(rem_ids) > 2500:
                selected = rng.choice(np.arange(len(rem_ids)), size=2500, replace=False)
                rem_ids = rem_ids[selected]
                rem_labels = rem_labels[selected]
            probs_all.append(self._predict_probs_arrays(words[:prefix_len], labels[:prefix_len].astype(np.float32), rem_ids, apply_centering=False))
            labels_all.append(rem_labels.astype(np.int32))
        if len(probs_all) == 0:
            return 0.0
        probs = np.concatenate(probs_all)
        labels = np.concatenate(labels_all)
        logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(probs, 1e-6, 1 - 1e-6))
        best_bias = 0.0
        best_score = -1.0
        for bias in np.linspace(-1.5, 1.5, 61):
            score = float(balanced_accuracy_score(labels, (logits + bias >= 0.0).astype(np.int32)))
            if score > best_score:
                best_score = score
                best_bias = float(bias)
        return best_bias

    def initialize_user_state(self, optional_user_metadata: dict[str, Any] | None = None) -> UserState:
        return UserState(payload={"observed_word_ids": np.zeros(0, dtype=np.int32), "observed_labels": np.zeros(0, dtype=np.float32)})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        return UserState(payload={"observed_word_ids": new_ids, "observed_labels": new_labels})

    def _predict_probs_arrays(self, obs_ids: np.ndarray, obs_labels: np.ndarray, candidate_ids: np.ndarray, apply_centering: bool) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        if len(candidate_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            obs_feat = torch.from_numpy(self.word_features[obs_ids]) if len(obs_ids) > 0 else torch.zeros((0, self.word_features.shape[1]), dtype=torch.float32)
            obs_lab = torch.from_numpy(obs_labels.astype(np.float32)) if len(obs_labels) > 0 else torch.zeros((0,), dtype=torch.float32)
            outputs: list[np.ndarray] = []
            chunk_size = 2048
            for start in range(0, len(candidate_ids), chunk_size):
                chunk = candidate_ids[start : start + chunk_size]
                target_feat = torch.from_numpy(self.word_features[chunk])
                logits = self._score_with_context(target_feat, obs_feat, obs_lab, chunk, obs_labels) + self.decision_logit_bias
                outputs.append(logits.detach().cpu().numpy().astype(np.float32))
            all_logits = np.concatenate(outputs)
            if apply_centering and abs(self.config.dynamic_centering_weight) > 1e-12:
                all_logits = all_logits - self.config.dynamic_centering_weight * float(np.median(all_logits))
            if apply_centering and abs(self.config.user_rate_centering_weight) > 1e-12 and len(obs_labels) > 0:
                user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0))
                quantile = float(np.clip(1.0 - user_rate, 0.05, 0.95))
                all_logits = all_logits - self.config.user_rate_centering_weight * float(np.quantile(all_logits, quantile))
            probs = 1.0 / (1.0 + np.exp(-all_logits))
            return np.clip(probs.astype(np.float32), 1e-6, 1 - 1e-6)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        obs_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        obs_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        return self._predict_probs_arrays(obs_ids, obs_labels, candidate_word_ids.astype(np.int32), apply_centering=True)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1.0 - p)
