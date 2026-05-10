#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vocab_benchmark.benchmark import (
    _build_fixed_query_sequence,
    _build_prior_uncertain_query_sequence,
    _build_user_discriminative_query_sequence,
    _evaluate_responses,
)
from vocab_benchmark.data import load_all
from vocab_benchmark.estimators.calibrated import (
    OnlineAccuracyThresholdCalibratedEstimator,
    BatchMedianCenteredEstimator,
    OnlineThresholdCalibratedEstimator,
    ProbabilityPowerCalibratedEstimator,
)
from vocab_benchmark.estimators.ensemble import BudgetAdaptiveEnsembleEstimator, WeightedAveragedEnsembleEstimator
from vocab_benchmark.estimators.fasttext_kernel import (
    FastTextKernelLogisticConfig,
    FastTextKernelLogisticEstimator,
    FastTextSVDRerankerConfig,
    FastTextSVDRerankerEstimator,
)
from vocab_benchmark.estimators.fasttext_semantic import FastTextSemanticConfig, FastTextSemanticPrototypeEstimator
from vocab_benchmark.estimators.irt import RaschIRTOnlineEstimator, TwoPLIRTOnlineEstimator
from vocab_benchmark.estimators.mf import LowRankMFOnlineEstimator
from vocab_benchmark.estimators.neural_advanced import NeuralMemoryMIRTConfig, NeuralMemoryMIRTEstimator
from vocab_benchmark.estimators.observed_user_vote import ObservedMatchUserVoteEstimator
from vocab_benchmark.estimators.online_user_logistic import OnlineUserLogisticEstimator
from vocab_benchmark.estimators.personalized import PersonalizedKNNPriorEstimator
from vocab_benchmark.estimators.svd import SVDRidgeUserEstimator
from vocab_benchmark.estimators.user_knn import UserKNNResponseEstimator
from vocab_benchmark.features import build_response_frame, build_word_feature_matrix, build_word_index
from vocab_benchmark.query_policies import UniformRandomPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-user', required=False)
    parser.add_argument('--all-test-users', action='store_true')
    parser.add_argument('--max-users', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=str, default='n2')
    parser.add_argument('--query-sequence', choices=['difficulty', 'user_discriminative', 'prior_uncertain'], default='difficulty')
    parser.add_argument('--budgets', type=str, default='50,100,200')
    parser.add_argument('--feature-set', choices=['legacy', 'fasttext_only', 'l2', 'freq', 'rich'], default='legacy')
    args = parser.parse_args()
    budgets = [int(part) for part in args.budgets.split(',') if part.strip()]
    if len(budgets) == 0:
        raise ValueError('budgets must contain at least one integer')
    sequence_len = max(max(budgets), 200)

    seed = 42
    loaded = load_all(Path('data'))
    responses = loaded.responses_static.copy()
    users = sorted(responses['user_id'].astype(str).unique().tolist())[: args.max_users]
    responses = responses[responses['user_id'].astype(str).isin(set(users))].copy()
    if args.all_test_users:
        test_users = users
    elif args.test_user is not None:
        test_users = [args.test_user]
    else:
        raise ValueError('either --test-user or --all-test-users is required')
    missing_test_users = sorted(set(test_users) - set(users))
    if missing_test_users:
        raise ValueError('test-user not in selected users')

    words = loaded.words
    x_words = build_word_feature_matrix(words, loaded.embeddings, loaded.frequency, feature_set=args.feature_set)
    word_index = build_word_index(words)
    user_index = {u: i for i, u in enumerate(sorted(responses['user_id'].astype(str).unique().tolist()))}
    resp_frame = build_response_frame(responses, user_index, word_index)
    if args.query_sequence == 'difficulty':
        fixed_query_sequence = _build_fixed_query_sequence(resp_frame, x_words, sequence_len, seed)
    elif args.query_sequence == 'user_discriminative':
        fixed_query_sequence = _build_user_discriminative_query_sequence(resp_frame, sequence_len, seed)
    else:
        fixed_query_sequence = _build_prior_uncertain_query_sequence(resp_frame, sequence_len, seed)

    configs = {
        'n2': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.6, user_rate_weight=1.2, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.0,
        ),
        'n2b': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.4, user_rate_weight=1.4, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.0,
        ),
        'n2c': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.0,
        ),
        'n2c_rate25': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.25,
        ),
        'n2c_rate15': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.15,
        ),
        'n2c_rate35': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.35,
        ),
        'n2c_rate50': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.5,
        ),
        'n2c_rate45': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=0.45,
        ),
        'n2c_rate100': NeuralMemoryMIRTConfig(
            hidden_dim=64, ability_dim=32, lr=1e-3, dropout=0.1, max_epochs=2, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=1,
            hard_negative_k=4, target_batch_size=96, min_prefix_len=8, max_prefix_len=64,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=1,
            user_rate_centering_weight=1.0,
        ),
        'n2c_rate35_e8': NeuralMemoryMIRTConfig(
            hidden_dim=128, ability_dim=64, lr=3e-4, dropout=0.2, max_epochs=8, seed=seed, weight_decay=1e-4,
            class_balance_weight=1.0, hard_negative_weight=0.2, balanced_surrogate_weight=0.2, early_stopping_patience=2,
            hard_negative_k=6, target_batch_size=128, min_prefix_len=16, max_prefix_len=128,
            prior_logit_weight=0.8, user_rate_weight=1.0, dynamic_centering_weight=1.0,
            warmup_epochs=2,
            user_rate_centering_weight=0.35,
        ),
    }
    fasttext_configs = {
        'ft_proto': FastTextSemanticConfig(
            embedding_dim=300, projection_dim=64, scalar_dim=16, hidden_dim=128,
            lr=1e-3, dropout=0.1, max_epochs=4, seed=seed, weight_decay=1e-4,
            target_batch_size=128, min_prefix_len=20, max_prefix_len=160,
            prior_logit_weight=0.8, user_rate_weight=1.0,
            dynamic_centering_weight=1.0, user_rate_centering_weight=0.35,
            memory_weight=0.5, prototype_weight=1.0,
            hard_negative_weight=0.15, hard_negative_k=6, early_stopping_patience=2,
        ),
        'ft_proto_light': FastTextSemanticConfig(
            embedding_dim=300, projection_dim=48, scalar_dim=12, hidden_dim=96,
            lr=1e-3, dropout=0.05, max_epochs=3, seed=seed, weight_decay=1e-4,
            target_batch_size=96, min_prefix_len=12, max_prefix_len=96,
            prior_logit_weight=0.85, user_rate_weight=1.0,
            dynamic_centering_weight=1.0, user_rate_centering_weight=0.35,
            memory_weight=0.35, prototype_weight=1.0,
            hard_negative_weight=0.1, hard_negative_k=4, early_stopping_patience=1,
        ),
        'ft_proto_deep': FastTextSemanticConfig(
            embedding_dim=300, projection_dim=96, scalar_dim=24, hidden_dim=160,
            lr=5e-4, dropout=0.2, max_epochs=6, seed=seed, weight_decay=2e-4,
            target_batch_size=160, min_prefix_len=24, max_prefix_len=200,
            prior_logit_weight=0.75, user_rate_weight=1.0,
            dynamic_centering_weight=1.0, user_rate_centering_weight=0.35,
            memory_weight=0.5, prototype_weight=1.2,
            hard_negative_weight=0.15, hard_negative_k=8, early_stopping_patience=2,
        ),
    }
    kernel_configs = {
        'ft_kernel': FastTextKernelLogisticConfig(
            embedding_dim=300, temperature=0.15, episodes_per_user=12, target_samples_per_episode=256,
            seed=seed, regularization_c=1.0, dynamic_centering_weight=1.0, user_rate_centering_weight=0.35,
        ),
        'ft_kernel_smooth': FastTextKernelLogisticConfig(
            embedding_dim=300, temperature=0.35, episodes_per_user=12, target_samples_per_episode=256,
            seed=seed, regularization_c=0.5, dynamic_centering_weight=1.0, user_rate_centering_weight=0.35,
        ),
    }
    reranker_configs = {
        'ft_svd_rerank': FastTextSVDRerankerConfig(
            kernel_config=kernel_configs['ft_kernel'],
            svd_rank=5, svd_ridge=1.0, svd_residual_scale=0.5, svd_intercept_ridge=1.0,
            regularization_c=0.5, dynamic_centering_weight=0.0, user_rate_centering_weight=0.0,
        ),
        'ft_svd_rerank_centered': FastTextSVDRerankerConfig(
            kernel_config=kernel_configs['ft_kernel'],
            svd_rank=5, svd_ridge=1.0, svd_residual_scale=0.5, svd_intercept_ridge=1.0,
            regularization_c=0.5, dynamic_centering_weight=0.5, user_rate_centering_weight=0.2,
        ),
    }
    known_static_configs = set(configs.keys()) | {
        *fasttext_configs.keys(),
        *kernel_configs.keys(),
        *reranker_configs.keys(),
        "n2c_hybrid",
        "n2c_rate35_svd_hybrid",
        "n2c_rate35_svd_hybrid_w30_70",
        "n2c_rate35_svd_hybrid_w70_30",
        "svd_ridge_r5_threshold",
        "svd_ridge_r5",
        "user_knn_k3",
        "user_knn_k5",
        "user_knn_k5_centered",
        "svd_ridge_r5_centered",
        "n2c_rate35_svd_userknn_hybrid",
        "ft_proto_svd_hybrid",
        "ft_proto_svd_hybrid_w30_70",
        "ft_kernel_svd_hybrid_w30_70",
        "n2c_svd_ftkernel_hybrid",
        "n2c_svd_ftkernel_hybrid_bias_p10",
        "n2c_svd_ftkernel_hybrid_bias_m10",
        "n2c_svd_ftkernel_hybrid_w20_55_25",
        "observed_vote_t05",
        "observed_vote_t10",
        "observed_vote_t20",
        "top_observed_vote_hybrid",
        "top_observed_vote_hybrid_w50_50",
        "observed_vote_t10_power2",
        "observed_vote_t10_power3",
        "online_user_logreg",
        "online_user_logreg_bal",
        "vote_userlogreg_hybrid",
        "top_vote_userlogreg_hybrid",
        "rasch_voteuser_hybrid_w90_10",
        "rasch_voteuser_hybrid_w85_15",
        "rasch_voteuser_hybrid_w75_25",
        "rasch_kernel_hybrid_w85_15",
        "rasch_voteuser_kernel_hybrid_w80_15_05",
        "twopl_voteuser_hybrid_w85_15",
        "rasch_twopl_voteuser_hybrid_w45_40_15",
        "refined_q100_raw",
        "refined_q1000_raw",
        "budget_adaptive_refined_raw",
        "rasch_highbudget",
        "rasch_highbudget_bias_p10",
        "rasch_highbudget_bias_p20",
        "rasch_highbudget_bias_m10",
        "twopl_highbudget",
        "vote_rasch_hybrid",
        "top_rasch_hybrid",
        "top_rasch_vote_hybrid",
    }
    if (
        args.config not in known_static_configs
        and not args.config.startswith("accuracy_threshold_rasch_twopl_voteuser_grid_")
        and not args.config.startswith("direct_rasch_twopl_vote_user_grid_")
        and not args.config.startswith("rasch_voteuser_grid_")
        and not args.config.startswith("rasch_twopl_voteuser_grid_")
    ):
        raise ValueError('unknown config')
    if args.config.startswith("direct_rasch_twopl_vote_user_grid_"):
        parts = args.config.removeprefix("direct_rasch_twopl_vote_user_grid_").split("_")
        if len(parts) != 5:
            raise ValueError(f"invalid direct_rasch_twopl_vote_user_grid config={args.config}")
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote, user_logreg],
            weights=[float(parts[0]) / 1000.0, float(parts[1]) / 1000.0, float(parts[2]) / 1000.0, float(parts[3]) / 1000.0],
            name=f"direct_rasch_twopl_vote_user_grid_w{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_b{parts[4]}",
            logit_bias=float(parts[4]) / 1000.0,
        )
    elif args.config.startswith("accuracy_threshold_rasch_twopl_voteuser_grid_"):
        parts = args.config.removeprefix("accuracy_threshold_rasch_twopl_voteuser_grid_").split("_")
        if len(parts) != 3:
            raise ValueError(f"invalid accuracy_threshold_rasch_twopl_voteuser_grid config={args.config}")
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        base = WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote_user],
            weights=[float(parts[0]) / 100.0, float(parts[1]) / 100.0, float(parts[2]) / 100.0],
            name=f"rasch_twopl_voteuser_grid_w{parts[0]}_{parts[1]}_{parts[2]}",
            logit_bias=0.0,
        )
        estimator = OnlineAccuracyThresholdCalibratedEstimator(
            base=base,
            min_observations=50,
            threshold_blend=1.0,
            name=f"accuracy_threshold_rasch_twopl_voteuser_grid_w{parts[0]}_{parts[1]}_{parts[2]}",
        )
    elif args.config.startswith("rasch_voteuser_grid_"):
        parts = args.config.removeprefix("rasch_voteuser_grid_").split("_")
        if len(parts) != 2:
            raise ValueError(f"invalid rasch_voteuser_grid config={args.config}")
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote_user],
            weights=[float(parts[0]) / 100.0, float(parts[1]) / 100.0],
            name=f"rasch_voteuser_grid_w{parts[0]}_{parts[1]}",
            logit_bias=0.0,
        )
    elif args.config.startswith("rasch_twopl_voteuser_grid_"):
        parts = args.config.removeprefix("rasch_twopl_voteuser_grid_").split("_")
        if len(parts) != 3:
            raise ValueError(f"invalid rasch_twopl_voteuser_grid config={args.config}")
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote_user],
            weights=[float(parts[0]) / 100.0, float(parts[1]) / 100.0, float(parts[2]) / 100.0],
            name=f"rasch_twopl_voteuser_grid_w{parts[0]}_{parts[1]}_{parts[2]}",
            logit_bias=0.0,
        )
    elif args.config == "n2c_hybrid":
        neural = NeuralMemoryMIRTEstimator(configs["n2c"])
        baseline = WeightedAveragedEnsembleEstimator(
            members=[PersonalizedKNNPriorEstimator(25, 1.0, 1.0, 0.4), LowRankMFOnlineEstimator(16, 3, 0.03, 0.01, seed)],
            weights=[0.7, 0.3],
            name="ensemble_knn25_mf16_w70_30",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, baseline],
            weights=[0.7, 0.3],
            name="neural_memory_mirt_n2c_hybrid",
            logit_bias=0.0,
        )
    elif args.config == "n2c_rate35_svd_hybrid":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.5, 0.5],
            name="neural_n2c_rate35_svd_r5_hybrid_w50_50",
            logit_bias=0.0,
        )
    elif args.config == "n2c_rate35_svd_hybrid_w30_70":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.3, 0.7],
            name="neural_n2c_rate35_svd_r5_hybrid_w30_70",
            logit_bias=0.0,
        )
    elif args.config == "n2c_rate35_svd_hybrid_w70_30":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.7, 0.3],
            name="neural_n2c_rate35_svd_r5_hybrid_w70_30",
            logit_bias=0.0,
        )
    elif args.config == "svd_ridge_r5":
        estimator = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        estimator.name = "svd_ridge_r5_l1_s05"
    elif args.config == "svd_ridge_r5_threshold":
        base = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        base.name = "svd_ridge_r5_l1_s05"
        estimator = OnlineThresholdCalibratedEstimator(
            base=base,
            min_observations=50,
            threshold_blend=0.5,
            name="svd_ridge_r5_l1_s05_online_threshold",
        )
    elif args.config == "user_knn_k3":
        estimator = UserKNNResponseEstimator(n_neighbors=3, prior_blend=0.35, similarity_temperature=0.35)
        estimator.name = "user_knn_response_k3_pb35_t35"
    elif args.config == "user_knn_k5":
        estimator = UserKNNResponseEstimator(n_neighbors=5, prior_blend=0.35, similarity_temperature=0.35)
        estimator.name = "user_knn_response_k5_pb35_t35"
    elif args.config == "user_knn_k5_centered":
        base = UserKNNResponseEstimator(n_neighbors=5, prior_blend=0.35, similarity_temperature=0.35)
        base.name = "user_knn_response_k5_pb35_t35"
        estimator = BatchMedianCenteredEstimator(base=base, center_weight=1.0, name="user_knn_response_k5_pb35_t35_centered")
    elif args.config == "svd_ridge_r5_centered":
        base = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        base.name = "svd_ridge_r5_l1_s05"
        estimator = BatchMedianCenteredEstimator(base=base, center_weight=0.5, name="svd_ridge_r5_l1_s05_centered_w50")
    elif args.config == "n2c_rate35_svd_userknn_hybrid":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        user_knn = UserKNNResponseEstimator(n_neighbors=3, prior_blend=0.35, similarity_temperature=0.35)
        user_knn.name = "user_knn_response_k3_pb35_t35"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, user_knn],
            weights=[0.25, 0.55, 0.20],
            name="neural_n2c_rate35_svd_r5_userknn_hybrid_w25_55_20",
            logit_bias=0.0,
        )
    elif args.config == "ft_proto_svd_hybrid":
        neural = FastTextSemanticPrototypeEstimator(fasttext_configs["ft_proto"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.5, 0.5],
            name="fasttext_proto_svd_r5_hybrid_w50_50",
            logit_bias=0.0,
        )
    elif args.config == "ft_proto_svd_hybrid_w30_70":
        neural = FastTextSemanticPrototypeEstimator(fasttext_configs["ft_proto"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.3, 0.7],
            name="fasttext_proto_svd_r5_hybrid_w30_70",
            logit_bias=0.0,
        )
    elif args.config == "ft_kernel_svd_hybrid_w30_70":
        neural = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd],
            weights=[0.3, 0.7],
            name="fasttext_kernel_svd_r5_hybrid_w30_70",
            logit_bias=0.0,
        )
    elif args.config == "n2c_svd_ftkernel_hybrid":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, kernel],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15",
            logit_bias=0.0,
        )
    elif args.config == "n2c_svd_ftkernel_hybrid_bias_p10":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, kernel],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15_bias_p10",
            logit_bias=0.1,
        )
    elif args.config == "n2c_svd_ftkernel_hybrid_bias_m10":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, kernel],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15_bias_m10",
            logit_bias=-0.1,
        )
    elif args.config == "n2c_svd_ftkernel_hybrid_w20_55_25":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, kernel],
            weights=[0.20, 0.55, 0.25],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w20_55_25",
            logit_bias=0.0,
        )
    elif args.config == "observed_vote_t05":
        estimator = ObservedMatchUserVoteEstimator(temperature=0.05, prior_blend=0.0, power=1.0)
        estimator.name = "observed_match_user_vote_t05"
    elif args.config == "observed_vote_t10":
        estimator = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        estimator.name = "observed_match_user_vote_t10"
    elif args.config == "observed_vote_t20":
        estimator = ObservedMatchUserVoteEstimator(temperature=0.20, prior_blend=0.0, power=1.0)
        estimator.name = "observed_match_user_vote_t20"
    elif args.config == "top_observed_vote_hybrid":
        neural = NeuralMemoryMIRTEstimator(configs["n2c_rate35"])
        svd = SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0)
        svd.name = "svd_ridge_r5_l1_s05"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"])
        observed_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        observed_vote.name = "observed_match_user_vote_t10"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[neural, svd, kernel, observed_vote],
            weights=[0.15, 0.35, 0.10, 0.40],
            name="top_observed_vote_hybrid_w15_35_10_40",
            logit_bias=0.0,
        )
    elif args.config == "top_observed_vote_hybrid_w50_50":
        top = WeightedAveragedEnsembleEstimator(
            members=[
                NeuralMemoryMIRTEstimator(configs["n2c_rate35"]),
                SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0),
                FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"]),
            ],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15",
            logit_bias=0.0,
        )
        observed_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        observed_vote.name = "observed_match_user_vote_t10"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[top, observed_vote],
            weights=[0.50, 0.50],
            name="top_observed_vote_hybrid_w50_50",
            logit_bias=0.0,
        )
    elif args.config == "observed_vote_t10_power2":
        base = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        base.name = "observed_match_user_vote_t10"
        estimator = ProbabilityPowerCalibratedEstimator(base=base, power=2.0, name="observed_match_user_vote_t10_power2")
    elif args.config == "observed_vote_t10_power3":
        base = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        base.name = "observed_match_user_vote_t10"
        estimator = ProbabilityPowerCalibratedEstimator(base=base, power=3.0, name="observed_match_user_vote_t10_power3")
    elif args.config == "online_user_logreg":
        estimator = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        estimator.name = "online_user_logreg_c01_pb50"
    elif args.config == "online_user_logreg_bal":
        estimator = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=True, min_observations=50)
        estimator.name = "online_user_logreg_c01_pb50_bal"
    elif args.config == "vote_userlogreg_hybrid":
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
    elif args.config == "top_vote_userlogreg_hybrid":
        top = WeightedAveragedEnsembleEstimator(
            members=[
                NeuralMemoryMIRTEstimator(configs["n2c_rate35"]),
                SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0),
                FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"]),
            ],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15",
            logit_bias=0.0,
        )
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[top, vote, user_logreg],
            weights=[0.35, 0.50, 0.15],
            name="top_vote_userlogreg_hybrid_w35_50_15",
            logit_bias=0.0,
        )
    elif args.config == "rasch_voteuser_hybrid_w90_10":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote_user],
            weights=[0.90, 0.10],
            name="rasch_voteuser_hybrid_w90_10",
            logit_bias=0.0,
        )
    elif args.config == "rasch_voteuser_hybrid_w85_15":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote_user],
            weights=[0.85, 0.15],
            name="rasch_voteuser_hybrid_w85_15",
            logit_bias=0.0,
        )
    elif args.config == "rasch_voteuser_hybrid_w75_25":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote_user],
            weights=[0.75, 0.25],
            name="rasch_voteuser_hybrid_w75_25",
            logit_bias=0.0,
        )
    elif args.config == "rasch_kernel_hybrid_w85_15":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel_smooth"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, kernel],
            weights=[0.85, 0.15],
            name="rasch_kernel_hybrid_w85_15",
            logit_bias=0.0,
        )
    elif args.config == "rasch_voteuser_kernel_hybrid_w80_15_05":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        kernel = FastTextKernelLogisticEstimator(kernel_configs["ft_kernel_smooth"])
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote, user_logreg, kernel],
            weights=[0.80, 0.1125, 0.0375, 0.05],
            name="rasch_voteuser_kernel_hybrid_w80_15_05",
            logit_bias=0.0,
        )
    elif args.config == "twopl_voteuser_hybrid_w85_15":
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[twopl, vote_user],
            weights=[0.85, 0.15],
            name="twopl_voteuser_hybrid_w85_15",
            logit_bias=0.0,
        )
    elif args.config == "rasch_twopl_voteuser_hybrid_w45_40_15":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        vote_user = WeightedAveragedEnsembleEstimator(
            members=[vote, user_logreg],
            weights=[0.75, 0.25],
            name="vote_userlogreg_hybrid_w75_25",
            logit_bias=0.0,
        )
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote_user],
            weights=[0.45, 0.40, 0.15],
            name="rasch_twopl_voteuser_hybrid_w45_40_15",
            logit_bias=0.0,
        )
    elif args.config == "refined_q100_raw":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, vote, user_logreg],
            weights=[0.76, 0.175, 0.065],
            name="refined_q100_raw_w760_175_065_bias_p020",
            logit_bias=0.020,
        )
    elif args.config == "refined_q1000_raw":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        twopl.name = "twopl_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        user_logreg = OnlineUserLogisticEstimator(regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50)
        user_logreg.name = "online_user_logreg_c01_pb50"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch, twopl, vote, user_logreg],
            weights=[0.29, 0.50, 0.13, 0.08],
            name="refined_q1000_raw_w290_500_130_080_bias_p015",
            logit_bias=0.015,
        )
    elif args.config == "budget_adaptive_refined_raw":
        low_rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        low_rasch.name = "rasch_highbudget_var25"
        low_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        low_vote.name = "observed_match_user_vote_t10"
        low_user_logreg = OnlineUserLogisticEstimator(
            regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50
        )
        low_user_logreg.name = "online_user_logreg_c01_pb50"
        low_budget = WeightedAveragedEnsembleEstimator(
            members=[low_rasch, low_vote, low_user_logreg],
            weights=[0.76, 0.175, 0.065],
            name="refined_q100_raw_w760_175_065_bias_p020",
            logit_bias=0.020,
        )
        high_rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        high_rasch.name = "rasch_highbudget_var25"
        high_twopl = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        high_twopl.name = "twopl_highbudget_var25"
        high_vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        high_vote.name = "observed_match_user_vote_t10"
        high_user_logreg = OnlineUserLogisticEstimator(
            regularization_c=0.1, prior_blend=0.5, class_weight_balanced=False, min_observations=50
        )
        high_user_logreg.name = "online_user_logreg_c01_pb50"
        high_budget = WeightedAveragedEnsembleEstimator(
            members=[high_rasch, high_twopl, high_vote, high_user_logreg],
            weights=[0.29, 0.50, 0.13, 0.08],
            name="refined_q1000_raw_w290_500_130_080_bias_p015",
            logit_bias=0.015,
        )
        estimator = BudgetAdaptiveEnsembleEstimator(
            low_budget=low_budget,
            high_budget=high_budget,
            switch_observations=500,
            name="budget_adaptive_refined_raw_switch500",
        )
    elif args.config == "rasch_highbudget":
        estimator = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        estimator.name = "rasch_highbudget_var25"
    elif args.config == "rasch_highbudget_bias_p10":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch],
            weights=[1.0],
            name="rasch_highbudget_var25_bias_p10",
            logit_bias=0.1,
        )
    elif args.config == "rasch_highbudget_bias_p20":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch],
            weights=[1.0],
            name="rasch_highbudget_var25_bias_p20",
            logit_bias=0.2,
        )
    elif args.config == "rasch_highbudget_bias_m10":
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[rasch],
            weights=[1.0],
            name="rasch_highbudget_var25_bias_m10",
            logit_bias=-0.1,
        )
    elif args.config == "twopl_highbudget":
        estimator = TwoPLIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        estimator.name = "twopl_highbudget_var25"
    elif args.config == "vote_rasch_hybrid":
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[vote, rasch],
            weights=[0.75, 0.25],
            name="vote_rasch_hybrid_w75_25",
            logit_bias=0.0,
        )
    elif args.config == "top_rasch_hybrid":
        top = WeightedAveragedEnsembleEstimator(
            members=[
                NeuralMemoryMIRTEstimator(configs["n2c_rate35"]),
                SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0),
                FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"]),
            ],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15",
            logit_bias=0.0,
        )
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[top, rasch],
            weights=[0.25, 0.75],
            name="top_rasch_hybrid_w25_75",
            logit_bias=0.0,
        )
    elif args.config == "top_rasch_vote_hybrid":
        top = WeightedAveragedEnsembleEstimator(
            members=[
                NeuralMemoryMIRTEstimator(configs["n2c_rate35"]),
                SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5, intercept_ridge=1.0),
                FastTextKernelLogisticEstimator(kernel_configs["ft_kernel"]),
            ],
            weights=[0.25, 0.60, 0.15],
            name="neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15",
            logit_bias=0.0,
        )
        rasch = RaschIRTOnlineEstimator(prior_var=25.0, lr=1.0, n_fit_steps=20)
        rasch.name = "rasch_highbudget_var25"
        vote = ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0, power=1.0)
        vote.name = "observed_match_user_vote_t10"
        estimator = WeightedAveragedEnsembleEstimator(
            members=[top, rasch, vote],
            weights=[0.20, 0.60, 0.20],
            name="top_rasch_vote_hybrid_w20_60_20",
            logit_bias=0.0,
        )
    elif args.config in kernel_configs:
        estimator = FastTextKernelLogisticEstimator(kernel_configs[args.config])
    elif args.config in reranker_configs:
        estimator = FastTextSVDRerankerEstimator(reranker_configs[args.config])
    elif args.config in fasttext_configs:
        estimator = FastTextSemanticPrototypeEstimator(fasttext_configs[args.config])
    else:
        estimator = NeuralMemoryMIRTEstimator(configs[args.config])

    out = _evaluate_responses(
        responses=responses,
        words=words,
        x_words=x_words,
        embedding_backend=loaded.embedding_backend,
        dataset_name='responses_static',
        data_mode=f'neural_split_eval_{args.query_sequence}',
        splits=[{'split_id': f's_{test_user}', 'test_user_id': test_user} for test_user in test_users],
        estimators=[estimator],
        policies=[UniformRandomPolicy()],
        budgets=budgets,
        rng=np.random.default_rng(seed),
        fixed_query_sequence=fixed_query_sequence,
        max_candidate_words_per_user=None,
    )
    out['model'] = f'neural_memory_mirt_{args.config}_{args.query_sequence}_{args.feature_set}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
