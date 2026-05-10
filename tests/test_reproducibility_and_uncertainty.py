from __future__ import annotations

import numpy as np

from vocab_benchmark.estimators.baselines import UserRateDifficultyEstimator


def test_uncertainty_decreases_with_observations() -> None:
    x = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    import pandas as pd
    rows = [{"user_idx": u, "word_idx": w, "label": int(w < 15)} for u in range(10) for w in range(30)]
    train = pd.DataFrame(rows)
    est = UserRateDifficultyEstimator(alpha=1.0, beta=1.0, blend=0.2)
    est.fit(train, x)
    st0 = est.initialize_user_state()
    cand = np.arange(30, dtype=np.int32)
    u0 = float(np.mean(est.predict_uncertainty(st0, cand)))
    st1 = est.update_user_state(st0, np.array([0, 1, 2, 3, 4], dtype=np.int32), np.array([1, 1, 1, 1, 1], dtype=np.int32))
    u1 = float(np.mean(est.predict_uncertainty(st1, cand)))
    assert u1 < u0


def test_reproducibility_rng() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = rng1.normal(size=10)
    b = rng2.normal(size=10)
    assert np.allclose(a, b)
