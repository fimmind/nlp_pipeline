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


class _MemoryMIRTModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, ability_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.ability_dim = ability_dim
        self.obs_proj = nn.Linear(feature_dim + 1, hidden_dim)
        self.obs_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRUCell(hidden_dim, ability_dim)
        self.discrimination = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ability_dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.key_proj = nn.Linear(feature_dim + 1, hidden_dim)
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.memory_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def init_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.ability_dim, device=device)

    def encode_observations(self, obs_feat: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        theta = self.init_theta(1, obs_feat.device)
        if obs_feat.shape[0] == 0:
            return theta
        for i in range(obs_feat.shape[0]):
            label_scalar = obs_labels[i].view(1, 1)
            x = torch.cat([obs_feat[i].view(1, -1), label_scalar], dim=-1)
            x = self.obs_dropout(torch.tanh(self.obs_proj(x)))
            theta = self.encoder(x, theta)
        return theta

    def _memory_residual(self, target_feat: torch.Tensor, obs_feat: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        if obs_feat.shape[0] == 0:
            return torch.zeros(target_feat.shape[0], device=target_feat.device)
        obs_label_signed = (2.0 * obs_labels - 1.0).view(-1, 1)
        mem_input = torch.cat([obs_feat, obs_label_signed], dim=-1)
        keys = torch.tanh(self.key_proj(mem_input))
        queries = torch.tanh(self.query_proj(target_feat))
        scale = float(np.sqrt(max(1, self.hidden_dim)))
        scores = (queries @ keys.transpose(0, 1)) / scale
        attn = torch.softmax(scores, dim=-1)
        residual = attn @ obs_label_signed
        return residual.squeeze(-1)

    def score(self, theta: torch.Tensor, target_feat: torch.Tensor, obs_feat: torch.Tensor, obs_labels: torch.Tensor) -> torch.Tensor:
        discr = self.discrimination(target_feat)
        base = torch.sum(discr * theta.repeat(target_feat.shape[0], 1), dim=-1) + self.bias(target_feat).squeeze(-1)
        memory = self._memory_residual(target_feat, obs_feat, obs_labels)
        return base + self.memory_scale * memory


@dataclass(frozen=True)
class NeuralMemoryMIRTConfig:
    hidden_dim: int
    ability_dim: int
    lr: float
    dropout: float
    max_epochs: int
    seed: int
    weight_decay: float
    class_balance_weight: float
    hard_negative_weight: float
    balanced_surrogate_weight: float
    early_stopping_patience: int
    hard_negative_k: int
    target_batch_size: int
    min_prefix_len: int
    max_prefix_len: int
    prior_logit_weight: float
    user_rate_weight: float
    dynamic_centering_weight: float
    warmup_epochs: int
    user_rate_centering_weight: float


class NeuralMemoryMIRTEstimator(Estimator):
    name = "neural_memory_mirt"

    def __init__(self, config: NeuralMemoryMIRTConfig) -> None:
        self.config = config
        self.model: _MemoryMIRTModule | None = None
        self.word_features = np.zeros((0, 0), dtype=np.float32)
        self.neighbors = np.empty((0, 0), dtype=np.int32)
        self.word_prior_logit = np.zeros(0, dtype=np.float32)
        self.decision_logit_bias = 0.0

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.word_features = word_features.astype(np.float32)
        self._build_neighbors()
        self._build_word_prior_logit(train_responses=train_responses)
        feature_dim = int(self.word_features.shape[1])
        self.model = _MemoryMIRTModule(feature_dim, self.config.hidden_dim, self.config.ability_dim, self.config.dropout)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.use_deterministic_algorithms(True)
        if train_responses.empty:
            return
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(self.config.seed)
        user_sequences = self._build_user_sequences(train_responses)
        device = torch.device("cpu")
        self.model.to(device)
        best_loss = float("inf")
        stale = 0
        for epoch in range(self.config.max_epochs):
            rng.shuffle(user_sequences)
            total_loss = 0.0
            total_steps = 0
            for words, labels in user_sequences:
                if len(words) < 8:
                    continue
                prefix_len = self._sample_prefix_len(rng, len(words), epoch)
                obs_words = words[:prefix_len]
                obs_labels = labels[:prefix_len]
                target_words = words[prefix_len:]
                target_labels = labels[prefix_len:]
                if len(target_words) == 0:
                    continue
                if len(target_words) > self.config.target_batch_size:
                    sel = rng.choice(np.arange(len(target_words)), size=self.config.target_batch_size, replace=False)
                    target_words = target_words[sel]
                    target_labels = target_labels[sel]
                obs_feat = torch.from_numpy(self.word_features[obs_words]).to(device)
                obs_lab = torch.from_numpy(obs_labels.astype(np.float32)).to(device)
                target_feat = torch.from_numpy(self.word_features[target_words]).to(device)
                y_true = torch.from_numpy(target_labels.astype(np.float32)).to(device)

                use_aux_losses = epoch >= self.config.warmup_epochs
                theta = self.model.encode_observations(obs_feat, obs_lab)
                logits = self.model.score(theta, target_feat, obs_feat, obs_lab)
                prior_logits = torch.from_numpy(self.word_prior_logit[target_words]).to(device)
                user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0))
                user_bias = self.config.user_rate_weight * (user_rate - 0.5)
                logits = logits + self.config.prior_logit_weight * prior_logits + user_bias

                pos_count = float(np.sum(target_labels == 1))
                neg_count = float(np.sum(target_labels == 0))
                pos_weight_value = (neg_count + 1.0) / (pos_count + 1.0)
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                bce = f.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pos_weight)

                p = torch.sigmoid(logits)
                pos_mask = y_true > 0.5
                neg_mask = y_true <= 0.5
                if int(pos_mask.sum().item()) > 0 and int(neg_mask.sum().item()) > 0:
                    ba_surrogate = 0.5 * ((1.0 - p[pos_mask]).mean() + p[neg_mask].mean())
                else:
                    ba_surrogate = torch.tensor(0.0, device=device)

                hard_neg = torch.tensor(0.0, device=device)
                if use_aux_losses and self.config.hard_negative_weight > 0.0:
                    hard_neg = self._hard_negative_loss(
                        rng=rng,
                        target_words=target_words,
                        target_labels=target_labels,
                        logits=logits,
                        theta=theta,
                        obs_feat=obs_feat,
                        obs_lab=obs_lab,
                        user_bias=user_bias,
                        device=device,
                    )
                bal_weight = self.config.balanced_surrogate_weight if use_aux_losses else 0.0
                hard_weight = self.config.hard_negative_weight if use_aux_losses else 0.0

                loss = (
                    self.config.class_balance_weight * bce
                    + bal_weight * ba_surrogate
                    + hard_weight * hard_neg
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_steps += 1
            if total_steps == 0:
                break
            mean_loss = total_loss / total_steps
            if mean_loss + 1e-6 < best_loss:
                best_loss = mean_loss
                stale = 0
            else:
                stale += 1
                if stale >= self.config.early_stopping_patience:
                    break
        self.decision_logit_bias = self._fit_decision_logit_bias(train_responses=train_responses, rng=rng)

    def _build_neighbors(self) -> None:
        n_words = self.word_features.shape[0]
        if n_words == 0:
            self.neighbors = np.empty((0, 0), dtype=np.int32)
            return
        k = min(max(1, self.config.hard_negative_k * 4), n_words - 1 if n_words > 1 else 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn_model.fit(self.word_features)
        idx = nn_model.kneighbors(return_distance=False)
        self.neighbors = idx[:, 1:].astype(np.int32)

    def _build_word_prior_logit(self, train_responses: pd.DataFrame) -> None:
        n_words = self.word_features.shape[0]
        prior = np.full(n_words, 0.5, dtype=np.float32)
        if not train_responses.empty:
            grp = train_responses.groupby("word_idx")["label"].agg(["sum", "count"])
            for idx, row in grp.iterrows():
                p = float((row["sum"] + 1.0) / (row["count"] + 2.0))
                prior[int(idx)] = p
        prior = np.clip(prior, 1e-5, 1 - 1e-5)
        self.word_prior_logit = np.log(prior) - np.log(1.0 - prior)

    def _build_user_sequences(self, train_responses: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        seqs: list[tuple[np.ndarray, np.ndarray]] = []
        grouped = train_responses.sort_values(["user_idx", "word_idx"]).groupby("user_idx")
        for _, g in grouped:
            words = g["word_idx"].to_numpy(dtype=np.int32)
            labels = g["label"].to_numpy(dtype=np.int32)
            seqs.append((words, labels))
        return seqs

    def _sample_prefix_len(self, rng: np.random.Generator, total_len: int, epoch: int) -> int:
        min_prefix = max(2, self.config.min_prefix_len)
        epoch_cap = int((epoch + 1) / max(1, self.config.max_epochs) * total_len)
        max_prefix = max(min_prefix + 1, min(self.config.max_prefix_len, epoch_cap if epoch_cap > 0 else min_prefix + 1))
        clipped_max = max(min_prefix + 1, min(total_len - 1, max_prefix))
        return int(rng.integers(min_prefix, clipped_max))

    def _hard_negative_loss(
        self,
        rng: np.random.Generator,
        target_words: np.ndarray,
        target_labels: np.ndarray,
        logits: torch.Tensor,
        theta: torch.Tensor,
        obs_feat: torch.Tensor,
        obs_lab: torch.Tensor,
        user_bias: float,
        device: torch.device,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        pos_positions = np.where(target_labels == 1)[0]
        if len(pos_positions) == 0:
            return torch.tensor(0.0, device=device)
        losses: list[torch.Tensor] = []
        target_map = {int(w): int(y) for w, y in zip(target_words.tolist(), target_labels.tolist())}
        for pos_pos in pos_positions.tolist()[: min(len(pos_positions), 16)]:
            pos_word = int(target_words[pos_pos])
            pos_logit = logits[pos_pos]
            cand = self.neighbors[pos_word].tolist() if self.neighbors.size > 0 else []
            neg_words = [w for w in cand if target_map.get(int(w), -1) == 0]
            if len(neg_words) == 0:
                continue
            neg_sel = np.array(neg_words[: self.config.hard_negative_k], dtype=np.int32)
            neg_feat = torch.from_numpy(self.word_features[neg_sel]).to(device)
            neg_logits = self.model.score(theta, neg_feat, obs_feat, obs_lab)
            neg_prior_logits = torch.from_numpy(self.word_prior_logit[neg_sel]).to(device)
            neg_logits = neg_logits + self.config.prior_logit_weight * neg_prior_logits + user_bias
            hard_neg = torch.max(neg_logits)
            margin = torch.relu(torch.tensor(1.0, device=device) - (pos_logit - hard_neg))
            losses.append(margin)
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.mean(torch.stack(losses))

    def _fit_decision_logit_bias(self, train_responses: pd.DataFrame, rng: np.random.Generator) -> float:
        if self.model is None or train_responses.empty:
            return 0.0
        users = sorted(train_responses["user_idx"].astype(int).unique().tolist())
        if len(users) == 0:
            return 0.0
        sampled_users = users[: min(6, len(users))]
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for user_idx in sampled_users:
            rows = train_responses[train_responses["user_idx"] == user_idx].sort_values("word_idx")
            words = rows["word_idx"].to_numpy(dtype=np.int32)
            labels = rows["label"].to_numpy(dtype=np.int32)
            if len(words) < 64:
                continue
            prefix_len = min(100, max(20, len(words) // 5))
            obs_ids = words[:prefix_len]
            obs_labels = labels[:prefix_len].astype(np.float32)
            rem_ids = words[prefix_len:]
            rem_labels = labels[prefix_len:]
            if len(rem_ids) > 2000:
                sel = rng.choice(np.arange(len(rem_ids)), size=2000, replace=False)
                rem_ids = rem_ids[sel]
                rem_labels = rem_labels[sel]
            probs = self._predict_probs_arrays(obs_ids=obs_ids, obs_labels=obs_labels, candidate_ids=rem_ids)
            all_probs.append(probs)
            all_labels.append(rem_labels.astype(np.int32))
        if len(all_probs) == 0:
            return 0.0
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        best_bias = 0.0
        best_ba = -1.0
        for bias in np.linspace(-1.0, 1.0, 41):
            p = np.clip(probs, 1e-6, 1 - 1e-6)
            logit = np.log(p) - np.log(1 - p)
            shifted = 1.0 / (1.0 + np.exp(-(logit + bias)))
            y_pred = (shifted >= 0.5).astype(np.int32)
            ba = float(balanced_accuracy_score(labels, y_pred))
            if ba > best_ba:
                best_ba = ba
                best_bias = float(bias)
        return best_bias

    def _predict_probs_arrays(self, obs_ids: np.ndarray, obs_labels: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        with torch.no_grad():
            obs_feat = torch.from_numpy(self.word_features[obs_ids]) if len(obs_ids) > 0 else torch.zeros((0, self.word_features.shape[1]), dtype=torch.float32)
            obs_lab = torch.from_numpy(obs_labels.astype(np.float32)) if len(obs_labels) > 0 else torch.zeros((0,), dtype=torch.float32)
            theta = self.model.encode_observations(obs_feat, obs_lab)
            target_feat = torch.from_numpy(self.word_features[candidate_ids])
            logits = self.model.score(theta, target_feat, obs_feat, obs_lab)
            prior_logits = torch.from_numpy(self.word_prior_logit[candidate_ids])
            user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0)) if len(obs_labels) > 0 else 0.5
            user_bias = self.config.user_rate_weight * (user_rate - 0.5)
            logits = logits + self.config.prior_logit_weight * prior_logits + user_bias
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            return np.clip(probs, 1e-6, 1 - 1e-6)

    def initialize_user_state(self, optional_user_metadata: dict[str, Any] | None = None) -> UserState:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        theta = self.model.init_theta(1, torch.device("cpu")).detach().cpu().numpy()
        payload: dict[str, Any] = {
            "theta": theta,
            "observed_word_ids": np.zeros(0, dtype=np.int32),
            "observed_labels": np.zeros(0, dtype=np.float32),
        }
        return UserState(payload=payload)

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        prev_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
        prev_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
        new_ids = np.concatenate([prev_ids, observed_word_ids.astype(np.int32)])
        new_labels = np.concatenate([prev_labels, observed_labels.astype(np.float32)])
        with torch.no_grad():
            obs_feat = torch.from_numpy(self.word_features[new_ids])
            obs_lab = torch.from_numpy(new_labels)
            theta = self.model.encode_observations(obs_feat, obs_lab).detach().cpu().numpy()
        payload: dict[str, Any] = {
            "theta": theta,
            "observed_word_ids": new_ids,
            "observed_labels": new_labels,
        }
        return UserState(payload=payload)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            theta = torch.from_numpy(np.asarray(user_state.payload["theta"], dtype=np.float32))
            obs_ids = np.asarray(user_state.payload["observed_word_ids"], dtype=np.int32)
            obs_labels = np.asarray(user_state.payload["observed_labels"], dtype=np.float32)
            obs_feat = torch.from_numpy(self.word_features[obs_ids]) if len(obs_ids) > 0 else torch.zeros((0, self.word_features.shape[1]), dtype=torch.float32)
            obs_lab = torch.from_numpy(obs_labels) if len(obs_labels) > 0 else torch.zeros((0,), dtype=torch.float32)

            outputs: list[np.ndarray] = []
            chunk_size = 1024
            for start in range(0, len(candidate_word_ids), chunk_size):
                chunk = candidate_word_ids[start : start + chunk_size]
                target_feat = torch.from_numpy(self.word_features[chunk])
                logits = self.model.score(theta, target_feat, obs_feat, obs_lab)
                prior_logits = torch.from_numpy(self.word_prior_logit[chunk])
                user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0)) if len(obs_labels) > 0 else 0.5
                user_bias = self.config.user_rate_weight * (user_rate - 0.5)
                logits = logits + self.config.prior_logit_weight * prior_logits + user_bias + self.decision_logit_bias
                outputs.append(logits.detach().cpu().numpy().astype(np.float32))
            all_logits = np.concatenate(outputs)
            if abs(self.config.dynamic_centering_weight) > 1e-12:
                median = float(np.median(all_logits))
                all_logits = all_logits - self.config.dynamic_centering_weight * median
            if abs(self.config.user_rate_centering_weight) > 1e-12 and len(obs_labels) > 0:
                user_rate = float((float(obs_labels.sum()) + 1.0) / (float(len(obs_labels)) + 2.0))
                threshold_quantile = float(np.clip(1.0 - user_rate, 0.05, 0.95))
                rate_center = float(np.quantile(all_logits, threshold_quantile))
                all_logits = all_logits - self.config.user_rate_centering_weight * rate_center
            all_probs = 1.0 / (1.0 + np.exp(-all_logits))
            return np.clip(all_probs, 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)
