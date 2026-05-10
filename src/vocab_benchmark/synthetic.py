from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class SyntheticDataset:
    words: pd.DataFrame
    frequency: pd.DataFrame
    embeddings: np.ndarray
    responses_static: pd.DataFrame


def generate_synthetic_dataset(n_words: int, n_users: int, embed_dim: int, rng: np.random.Generator) -> SyntheticDataset:
    embeddings = rng.normal(0.0, 1.0, size=(n_words, embed_dim)).astype(np.float32)
    word_ids = [f"w_{i}" for i in range(n_words)]
    words = pd.DataFrame(
        {
            "word_id": word_ids,
            "word": [f"word_{i}" for i in range(n_words)],
            "lemma": [f"word_{i}" for i in range(n_words)],
            "pos": ["X"] * n_words,
            "morphology": [""] * n_words,
            "language": ["en"] * n_words,
            "source": ["synthetic"] * n_words,
            "length": [6] * n_words,
        }
    )
    freq = rng.lognormal(mean=1.5, sigma=1.0, size=n_words)
    frequency = pd.DataFrame(
        {
            "word_id": word_ids,
            "word": words["word"],
            "language": ["en"] * n_words,
            "frequency": freq,
            "log_frequency": np.log1p(freq),
        }
    )
    b = 0.8 * rng.normal(0.0, 1.0, size=n_words) - 0.3 * (frequency["log_frequency"].to_numpy() - frequency["log_frequency"].mean())
    rows = []
    for u in range(n_users):
        theta = rng.normal(0.0, 1.0)
        p = expit(theta - b)
        y = rng.binomial(1, p)
        for i in range(n_words):
            rows.append(
                {
                    "user_id": f"u_{u}",
                    "word_id": word_ids[i],
                    "word": f"word_{i}",
                    "label": int(y[i]),
                    "raw_score": np.nan,
                    "timestamp": np.nan,
                    "source": "synthetic",
                    "language": "en",
                }
            )
    responses = pd.DataFrame(rows)
    return SyntheticDataset(words=words, frequency=frequency, embeddings=embeddings, responses_static=responses)
