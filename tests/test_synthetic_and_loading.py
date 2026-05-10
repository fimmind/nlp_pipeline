from __future__ import annotations

from pathlib import Path

import numpy as np

from vocab_benchmark.synthetic import generate_synthetic_dataset


def test_synthetic_shapes() -> None:
    ds = generate_synthetic_dataset(n_words=200, n_users=50, embed_dim=24, rng=np.random.default_rng(0))
    assert ds.embeddings.shape == (200, 24)
    assert len(ds.words) == 200
    assert len(ds.responses_static) == 200 * 50
    assert set(ds.responses_static["label"].unique().tolist()) <= {0, 1}


def test_prepare_outputs_exist() -> None:
    base = Path("data/processed")
    assert (base / "responses_static.csv").exists()
    assert (base / "words.csv").exists()
    assert (base / "embeddings.npy").exists()
