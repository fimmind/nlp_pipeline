#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vocab_benchmark.benchmark import _build_fixed_query_sequence, _evaluate_responses
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.ensemble import WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.mf import LowRankMFOnlineEstimator
from vocab_benchmark.estimators.personalized import PersonalizedKNNPriorEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-user', required=True)
    parser.add_argument('--max-users', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    seed = 42
    loaded = load_all(Path('data'))
    responses = loaded.responses_static.copy()
    users = sorted(responses['user_id'].astype(str).unique().tolist())[: args.max_users]
    responses = responses[responses['user_id'].astype(str).isin(set(users))].copy()
    if args.test_user not in set(users):
        raise ValueError('test-user not in selected users')

    words = loaded.words
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency)
    word_index = build_word_index(words)
    user_index = {u: i for i, u in enumerate(sorted(responses['user_id'].astype(str).unique().tolist()))}
    resp_frame = build_response_frame(responses, user_index, word_index)
    fixed_query_sequence = _build_fixed_query_sequence(resp_frame, x_words, 200, seed)

    estimator = WeightedAveragedEnsembleEstimator(
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
        data_mode='baseline_split_eval',
        splits=[{'split_id': f's_{args.test_user}', 'test_user_id': args.test_user}],
        estimators=[estimator],
        policies=[UniformRandomPolicy()],
        budgets=[50, 100, 200],
        rng=np.random.default_rng(seed),
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out['model'] = 'ensemble_knn25_mf16_w70_30'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
