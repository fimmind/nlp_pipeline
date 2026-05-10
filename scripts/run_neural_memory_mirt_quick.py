#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _build_loou_splits, _evaluate_responses, _extract_split_rows
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.ensemble import WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.mf import LowRankMFOnlineEstimator
from vocab_benchmark.estimators.neural_advanced import NeuralMemoryMIRTConfig, NeuralMemoryMIRTEstimator
from vocab_benchmark.estimators.personalized import PersonalizedKNNPriorEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def main() -> None:
    seed = 42
    max_users = 4
    out_dir = Path('/tmp/neural_memory_mirt_quick_u4')
    rng = np.random.default_rng(seed)
    loaded = load_all(Path('data'))
    words = loaded.words
    frequency = loaded.frequency
    embeddings = loaded.embeddings
    responses = loaded.responses_static.copy()

    keep = set(sorted(responses['user_id'].astype(str).unique().tolist())[:max_users])
    responses = responses[responses['user_id'].astype(str).isin(keep)].copy()

    x_words = build_word_feature_matrix(words, embeddings, frequency)
    word_index = build_word_index(words)
    user_index = {u: i for i, u in enumerate(sorted(responses['user_id'].astype(str).unique().tolist()))}
    resp_frame = build_response_frame(responses, user_index, word_index)
    fixed_query_sequence = _build_fixed_query_sequence(resp_frame, x_words, 200, seed)

    splits = _extract_split_rows(loaded.splits['static_leave_one_user_out']) if 'static_leave_one_user_out' in loaded.splits else _build_loou_splits(sorted(keep))
    splits = [s for s in splits if str(s['test_user_id']) in keep]

    neural = NeuralMemoryMIRTEstimator(
        NeuralMemoryMIRTConfig(
            hidden_dim=64,
            ability_dim=32,
            lr=5e-4,
            dropout=0.2,
            max_epochs=4,
            seed=seed,
            weight_decay=1e-4,
            class_balance_weight=1.0,
            hard_negative_weight=0.3,
            balanced_surrogate_weight=0.5,
            early_stopping_patience=2,
            hard_negative_k=6,
            target_batch_size=256,
            min_prefix_len=16,
            max_prefix_len=128,
            prior_logit_weight=0.8,
            user_rate_weight=1.0,
            dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.0,
        )
    )
    baseline = WeightedAveragedEnsembleEstimator(
        members=[PersonalizedKNNPriorEstimator(25, 1.0, 1.0, 0.4), LowRankMFOnlineEstimator(16, 3, 0.03, 0.01, seed)],
        weights=[0.7, 0.3],
        name='ensemble_knn25_mf16_w70_30',
        logit_bias=0.0,
    )

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name='responses_static',
        data_mode='neural_memory_mirt_quick',
        splits=splits,
        estimators=[neural, baseline],
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=rng,
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / 'results.csv', index=False)
    lb = out.groupby(['estimator', 'q'], as_index=False)[['balanced_accuracy', 'accuracy', 'nll', 'brier', 'auroc']].mean()
    lb.to_csv(out_dir / 'leaderboard.csv', index=False)
    print(lb.sort_values(['q', 'balanced_accuracy'], ascending=[True, False]).to_string(index=False))


if __name__ == '__main__':
    main()
