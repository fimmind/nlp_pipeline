from __future__ import annotations

import numpy as np
import pandas as pd

from vocab_benchmark.estimators.baselines import DifficultyStratifiedBetaEstimator, GlobalWordPriorEstimator, UserRateDifficultyEstimator
from vocab_benchmark.estimators.calibrated import OnlineThresholdCalibratedEstimator, OnlineThresholdWordPriorEstimator
from vocab_benchmark.estimators.collaborative import UserSimilarityKNNOnlineEstimator
from vocab_benchmark.estimators.irt import RaschIRTOnlineEstimator
from vocab_benchmark.estimators.neural import NeuralEncoderDecoderEstimator, NeuralEstimatorConfig
from vocab_benchmark.estimators.neural_advanced import NeuralMemoryMIRTConfig, NeuralMemoryMIRTEstimator
from vocab_benchmark.estimators.svd import SVDRidgeUserEstimator
from vocab_benchmark.metrics import classification_metrics
from vocab_benchmark.query_policies import EntropyPolicy, UniformRandomPolicy


def _toy_train() -> tuple[pd.DataFrame, np.ndarray]:
    x = np.random.default_rng(1).normal(size=(20, 6)).astype(np.float32)
    rows = []
    for u in range(6):
        for w in range(20):
            rows.append({"user_idx": u, "word_idx": w, "label": int((u + w) % 2 == 0)})
    return pd.DataFrame(rows), x


def test_estimator_interface_and_update() -> None:
    train, x = _toy_train()
    estimators = [
        GlobalWordPriorEstimator(alpha=1.0, beta=1.0),
        OnlineThresholdWordPriorEstimator(min_observations=3, threshold_blend=0.5, temperature=1.0),
        OnlineThresholdCalibratedEstimator(
            base=GlobalWordPriorEstimator(alpha=1.0, beta=1.0),
            min_observations=3,
            threshold_blend=0.5,
            name="threshold_calibrated_global_prior",
        ),
        UserRateDifficultyEstimator(alpha=1.0, beta=1.0, blend=0.5),
        DifficultyStratifiedBetaEstimator(alpha=1.0, beta=1.0, n_bins=4),
        UserSimilarityKNNOnlineEstimator(
            n_neighbors=3,
            shrinkage=5.0,
            prior_weight=0.1,
            temperature=1.0,
            logit_bias=0.0,
            rate_centering_weight=0.25,
            rate_alpha=2.0,
            rate_beta=2.0,
        ),
        RaschIRTOnlineEstimator(prior_var=4.0, lr=1.0, n_fit_steps=3),
        SVDRidgeUserEstimator(rank=3, ridge=1.0, residual_scale=1.0, intercept_ridge=1.0),
        NeuralEncoderDecoderEstimator(
            NeuralEstimatorConfig(
                architecture="gru_mlp",
                strategy="teacher_forced",
                hidden_dim=32,
                lr=1e-3,
                dropout=0.0,
                max_epochs=1,
                seed=1,
                weight_decay=1e-4,
                calibration_weight=0.0,
                early_stopping_patience=1,
            )
        ),
        NeuralMemoryMIRTEstimator(
            NeuralMemoryMIRTConfig(
                hidden_dim=32,
                ability_dim=16,
                lr=1e-3,
                dropout=0.0,
                max_epochs=1,
                seed=1,
                weight_decay=1e-4,
                class_balance_weight=1.0,
                hard_negative_weight=0.1,
                balanced_surrogate_weight=0.1,
                early_stopping_patience=1,
                hard_negative_k=2,
                target_batch_size=16,
                min_prefix_len=2,
                max_prefix_len=8,
                prior_logit_weight=0.5,
                user_rate_weight=0.5,
                dynamic_centering_weight=0.0,
                warmup_epochs=0,
                user_rate_centering_weight=0.0,
            )
        ),
    ]
    cand = np.arange(20, dtype=np.int32)
    for est in estimators:
        est.fit(train, x)
        state = est.initialize_user_state()
        p0 = est.predict_proba(state, cand)
        state = est.update_user_state(state, np.array([0, 1, 2], dtype=np.int32), np.array([1, 1, 0], dtype=np.int32))
        p1 = est.predict_proba(state, cand)
        assert p0.shape == (20,)
        assert p1.shape == (20,)
        assert np.all((p1 > 0) & (p1 < 1))


def test_query_policy_never_returns_queried() -> None:
    train, x = _toy_train()
    est = GlobalWordPriorEstimator(alpha=1.0, beta=1.0)
    est.fit(train, x)
    state = est.initialize_user_state()
    cand = np.arange(20, dtype=np.int32)
    queried = {1, 2, 3}
    policy = UniformRandomPolicy()
    q = policy.select_next_queries(est, state, cand, queried, batch_size=5, rng=np.random.default_rng(0))
    assert all(int(w) not in queried for w in q.tolist())


def test_metrics_simple_case() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    m = classification_metrics(y, p)
    assert m["nll"] < 0.4
    assert m["brier"] < 0.1
    assert m["auroc"] > 0.9


def test_entropy_policy_targets_uncertain() -> None:
    train, x = _toy_train()
    est = UserRateDifficultyEstimator(alpha=1.0, beta=1.0, blend=0.0)
    est.fit(train, x)
    st = est.initialize_user_state()
    pol = EntropyPolicy()
    q = pol.select_next_queries(est, st, np.arange(20, dtype=np.int32), set(), batch_size=3, rng=np.random.default_rng(0))
    assert len(q) == 3
