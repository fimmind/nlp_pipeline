#!/usr/bin/env python3
"""Export per-word prior probabilities from CLI estimators for use in the static site."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
# Ensure scripts/ is on path so we can import vocab_book_cli
sys.path.insert(0, str(repo_root / "scripts"))

from vocab_benchmark.data import load_all
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_book_cli import build_estimator, dataset_fingerprint


def main() -> None:
    data_dir = repo_root / "data"
    out_dir = repo_root / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    loaded = load_all(data_dir)
    words_df = loaded.words
    word_index = build_word_index(words_df)
    x_words = build_word_feature_matrix(words_df, loaded.embeddings, loaded.frequency, feature_set="rich")
    user_ids = sorted(loaded.responses_static["user_id"].astype(str).unique().tolist())
    user_index = {u: i for i, u in enumerate(user_ids)}
    resp_frame = build_response_frame(loaded.responses_static, user_index, word_index)
    data_fp = dataset_fingerprint(data_dir)
    print(f"Dataset fingerprint: {data_fp}")

    n_words = x_words.shape[0]
    word_list = [str(words_df.iloc[i]["word"]) for i in range(n_words)]

    # Build query pool from response frame (most observed words first)
    word_obs_counts = resp_frame.groupby("word_idx")["label"].count().to_dict()
    query_pool = sorted(
        [int(w) for w in word_obs_counts.keys()],
        key=lambda w: -word_obs_counts.get(w, 0),
    )
    query_pool_words = [word_list[w] for w in query_pool if 0 <= w < n_words]
    query_pool_words = query_pool_words[:2000]  # Cap for site size
    print(f"Query pool size: {len(query_pool_words)}")

    models_to_export = {
        "rasch": "rasch",
        "best_adaptive": "best_adaptive",
        "best_grouped_irt_model": "best_grouped_irt_model",
        "best_high_budget": "best_high_budget",
        "twopl": "twopl",
        "rasch_twopl_vote_user": "rasch_twopl_vote_user",
    }

    for model_key, filename in models_to_export.items():
        print(f"Fitting and exporting {model_key} ...")
        estimator = build_estimator(model_key, seed=42)
        estimator.fit(resp_frame, x_words)
        state = estimator.initialize_user_state()
        state = estimator.update_user_state(
            state, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        )
        probs = estimator.predict_proba(state, np.arange(n_words, dtype=np.int32))

        payload = {
            "model_key": model_key,
            "model_name": estimator.name,
            "words": word_list,
            "accuracy": [float(p) for p in probs],
            "query_pool": query_pool_words,
        }
        out_path = out_dir / f"{filename}_model_data.json"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"  Saved {out_path} ({len(word_list)} words, {out_path.stat().st_size / 1024:.1f} KB)")

    print("Done.")


if __name__ == "__main__":
    main()
