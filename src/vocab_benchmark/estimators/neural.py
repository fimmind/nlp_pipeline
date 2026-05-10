from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f

from .base import Estimator, UserState

Architecture = Literal["gru_mlp", "lstm_bilinear", "residual_gru_gated"]
Strategy = Literal["teacher_forced", "curriculum_prefix", "contrastive_hard_negative"]


class _EncoderDecoderModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, architecture: Architecture, dropout: float) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(feature_dim + 1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if architecture == "gru_mlp":
            self.cell = nn.GRUCell(hidden_dim, hidden_dim)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim + feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif architecture == "lstm_bilinear":
            self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
            self.word_proj = nn.Linear(feature_dim, hidden_dim)
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        else:
            self.cell = nn.GRUCell(hidden_dim, hidden_dim)
            self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gate = nn.Linear(feature_dim, hidden_dim)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim + feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        if self.architecture == "lstm_bilinear":
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            return h, c
        return h, None

    def update_step(
        self,
        h: torch.Tensor,
        c: torch.Tensor | None,
        word_feat: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.cat([word_feat, label], dim=-1)
        x = self.dropout(torch.tanh(self.input_proj(x)))
        if self.architecture == "lstm_bilinear":
            assert c is not None
            h_next, c_next = self.cell(x, (h, c))
            return h_next, c_next
        h_next = self.cell(x, h)
        if self.architecture == "residual_gru_gated":
            h_next = h_next + self.residual_proj(h)
        return h_next, c

    def decode_logits(self, h: torch.Tensor, word_feat: torch.Tensor) -> torch.Tensor:
        if self.architecture == "lstm_bilinear":
            w = torch.tanh(self.word_proj(word_feat))
            return self.bilinear(h, w).squeeze(-1)
        if self.architecture == "residual_gru_gated":
            gate = torch.sigmoid(self.gate(word_feat))
            gated_h = h * gate
            return self.decoder(torch.cat([gated_h, word_feat], dim=-1)).squeeze(-1)
        return self.decoder(torch.cat([h, word_feat], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class NeuralEstimatorConfig:
    architecture: Architecture
    strategy: Strategy
    hidden_dim: int
    lr: float
    dropout: float
    max_epochs: int
    seed: int
    weight_decay: float
    calibration_weight: float
    early_stopping_patience: int


class NeuralEncoderDecoderEstimator(Estimator):
    def __init__(self, config: NeuralEstimatorConfig) -> None:
        self.config = config
        self.name = f"neural_{config.architecture}_{config.strategy}_h{config.hidden_dim}_lr{config.lr}_do{config.dropout}_s{config.seed}"
        self.model: _EncoderDecoderModel | None = None
        self.word_features = np.zeros((0, 0), dtype=np.float32)

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        self.word_features = word_features.astype(np.float32)
        feature_dim = int(word_features.shape[1])
        self.model = _EncoderDecoderModel(
            feature_dim=feature_dim,
            hidden_dim=self.config.hidden_dim,
            architecture=self.config.architecture,
            dropout=self.config.dropout,
        )
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
        best_epoch_loss = float("inf")
        stale_epochs = 0
        for epoch in range(self.config.max_epochs):
            rng.shuffle(user_sequences)
            epoch_loss = 0.0
            steps = 0
            for words, labels in user_sequences:
                if len(words) < 2:
                    continue
                prefix_len = self._sample_prefix_len(rng=rng, total_len=len(words), epoch=epoch)
                h, c = self.model.init_state(1, device)
                for i in range(prefix_len):
                    wf = torch.from_numpy(self.word_features[words[i]]).to(device).reshape(1, -1)
                    y = torch.tensor([[float(labels[i])]], dtype=torch.float32, device=device)
                    h, c = self.model.update_step(h, c, wf, y)
                target_idx = words[prefix_len:]
                if len(target_idx) == 0:
                    continue
                target_labels = labels[prefix_len:]
                feat = torch.from_numpy(self.word_features[target_idx]).to(device)
                logits = self.model.decode_logits(h.repeat(len(target_idx), 1), feat)
                y_true = torch.from_numpy(target_labels.astype(np.float32)).to(device)
                loss = f.binary_cross_entropy_with_logits(logits, y_true)
                if self.config.strategy == "contrastive_hard_negative":
                    contrastive = self._contrastive_hard_negative_loss(logits=logits, labels=y_true)
                    loss = loss + 0.25 * contrastive
                if self.config.calibration_weight > 0.0:
                    p_mean = torch.sigmoid(logits).mean()
                    y_mean = y_true.mean()
                    calibration_proxy = (p_mean - y_mean) ** 2
                    loss = loss + self.config.calibration_weight * calibration_proxy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
            if steps == 0:
                break
            mean_epoch_loss = epoch_loss / steps
            if mean_epoch_loss + 1e-6 < best_epoch_loss:
                best_epoch_loss = mean_epoch_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break

    def _contrastive_hard_negative_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positives = labels > 0.5
        negatives = labels <= 0.5
        if int(positives.sum().item()) == 0 or int(negatives.sum().item()) == 0:
            return f.binary_cross_entropy_with_logits(logits, labels)
        positive_logits = logits[positives]
        negative_logits = logits[negatives]
        positive_hard = positive_logits[torch.argmin(positive_logits)]
        negative_hard = negative_logits[torch.argmax(negative_logits)]
        margin = 1.0
        return torch.relu(margin - (positive_hard - negative_hard))

    def _build_user_sequences(self, train_responses: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        seqs: list[tuple[np.ndarray, np.ndarray]] = []
        grouped = train_responses.sort_values(["user_idx", "word_idx"]).groupby("user_idx")
        for _, g in grouped:
            words = g["word_idx"].to_numpy(dtype=np.int32)
            labels = g["label"].to_numpy(dtype=np.int32)
            seqs.append((words, labels))
        return seqs

    def _sample_prefix_len(self, rng: np.random.Generator, total_len: int, epoch: int) -> int:
        if self.config.strategy == "teacher_forced":
            return int(rng.integers(1, total_len))
        if self.config.strategy == "curriculum_prefix":
            max_prefix = max(2, int((epoch + 1) / max(1, self.config.max_epochs) * total_len))
            return int(rng.integers(1, max_prefix))
        return int(rng.integers(1, total_len))

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        h, c = self.model.init_state(1, torch.device("cpu"))
        payload: dict[str, np.ndarray | str] = {"h": h.detach().cpu().numpy()}
        if c is not None:
            payload["c"] = c.detach().cpu().numpy()
        return UserState(payload=payload)

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        h = torch.from_numpy(np.asarray(user_state.payload["h"], dtype=np.float32))
        c = None
        if "c" in user_state.payload:
            c = torch.from_numpy(np.asarray(user_state.payload["c"], dtype=np.float32))
        for word_id, label in zip(observed_word_ids.tolist(), observed_labels.tolist()):
            wf = torch.from_numpy(self.word_features[int(word_id)]).reshape(1, -1)
            y = torch.tensor([[float(label)]], dtype=torch.float32)
            h, c = self.model.update_step(h, c, wf, y)
        payload: dict[str, np.ndarray] = {"h": h.detach().cpu().numpy()}
        if c is not None:
            payload["c"] = c.detach().cpu().numpy()
        return UserState(payload=payload)

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        if len(candidate_word_ids) == 0:
            return np.zeros(0, dtype=np.float32)
        h = torch.from_numpy(np.asarray(user_state.payload["h"], dtype=np.float32))
        feat = torch.from_numpy(self.word_features[candidate_word_ids])
        logits = self.model.decode_logits(h.repeat(len(candidate_word_ids), 1), feat)
        proba = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        return np.clip(proba, 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)


class AveragedEnsembleEstimator(Estimator):
    def __init__(self, members: list[Estimator], name: str, temperature: float) -> None:
        self.members = members
        self.name = name
        self.temperature = temperature

    def fit(self, train_responses: pd.DataFrame, word_features: np.ndarray) -> None:
        for member in self.members:
            member.fit(train_responses, word_features)

    def initialize_user_state(self, optional_user_metadata: dict | None = None) -> UserState:
        states = [member.initialize_user_state(optional_user_metadata) for member in self.members]
        return UserState(payload={"states": states})

    def update_user_state(self, user_state: UserState, observed_word_ids: np.ndarray, observed_labels: np.ndarray) -> UserState:
        states = user_state.payload["states"]
        new_states = []
        for member, state in zip(self.members, states):
            new_states.append(member.update_user_state(state, observed_word_ids, observed_labels))
        return UserState(payload={"states": new_states})

    def predict_proba(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        probs = []
        states = user_state.payload["states"]
        for member, state in zip(self.members, states):
            p = member.predict_proba(state, candidate_word_ids)
            if self.temperature != 1.0:
                logit = np.log(p) - np.log(1 - p)
                p = 1.0 / (1.0 + np.exp(-logit / self.temperature))
            probs.append(p)
        stacked = np.stack(probs, axis=0)
        return np.clip(np.mean(stacked, axis=0), 1e-6, 1 - 1e-6)

    def predict_uncertainty(self, user_state: UserState, candidate_word_ids: np.ndarray) -> np.ndarray:
        p = self.predict_proba(user_state, candidate_word_ids)
        return p * (1 - p)
