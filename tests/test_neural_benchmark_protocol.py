from __future__ import annotations

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _evaluate_responses
from vocab_benchmark.estimators.neural import NeuralEncoderDecoderEstimator, NeuralEstimatorConfig
from vocab_benchmark.query_policies import UniformRandomPolicy


def _toy_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n_words = 40
    n_users = 8
    words = pd.DataFrame({"word_id": [f"w{i}" for i in range(n_words)], "length": rng.integers(3, 9, size=n_words)})
    x_words = rng.normal(size=(n_words, 10)).astype(np.float32)
    rows = []
    for u in range(n_users):
        for w in range(n_words):
            rows.append({"user_id": f"u{u}", "word_id": f"w{w}", "label": int((u + w) % 3 != 0)})
    responses = pd.DataFrame(rows)
    return responses, words, x_words


def test_fixed_sequence_is_deterministic() -> None:
    responses, words, x_words = _toy_data()
    user_index = {u: i for i, u in enumerate(sorted(responses["user_id"].unique().tolist()))}
    word_index = {w: i for i, w in enumerate(words["word_id"].tolist())}
    resp = responses.copy()
    resp["user_idx"] = resp["user_id"].map(user_index)
    resp["word_idx"] = resp["word_id"].map(word_index)
    seq1 = _build_fixed_query_sequence(resp=resp[["user_idx", "word_idx", "label"]], x_words=x_words, sequence_len=20, seed=123)
    seq2 = _build_fixed_query_sequence(resp=resp[["user_idx", "word_idx", "label"]], x_words=x_words, sequence_len=20, seed=123)
    assert np.array_equal(seq1, seq2)


def test_neural_fixed_protocol_smoke() -> None:
    responses, words, x_words = _toy_data()
    user_index = {u: i for i, u in enumerate(sorted(responses["user_id"].unique().tolist()))}
    word_index = {w: i for i, w in enumerate(words["word_id"].tolist())}
    resp = responses.copy()
    resp["user_idx"] = resp["user_id"].map(user_index)
    resp["word_idx"] = resp["word_id"].map(word_index)
    seq = _build_fixed_query_sequence(resp=resp[["user_idx", "word_idx", "label"]], x_words=x_words, sequence_len=20, seed=7)
    est = NeuralEncoderDecoderEstimator(
        NeuralEstimatorConfig(
            architecture="residual_gru_gated",
            strategy="contrastive_hard_negative",
            hidden_dim=32,
            lr=1e-3,
            dropout=0.0,
            max_epochs=1,
            seed=7,
            weight_decay=1e-4,
            calibration_weight=0.0,
            early_stopping_patience=1,
        )
    )
    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend="synthetic",
        dataset_name="toy",
        data_mode="static_fixed200_neural",
        splits=[{"split_id": "s0", "test_user_id": "u0"}],
        estimators=[est],
        policies=[UniformRandomPolicy()],
        budgets=[5, 10],
        rng=np.random.default_rng(0),
        fixed_query_sequence=seq,
    )
    assert not out.empty
    assert np.all(out["balanced_accuracy"].to_numpy() >= 0.0)
    assert np.all(out["balanced_accuracy"].to_numpy() <= 1.0)
