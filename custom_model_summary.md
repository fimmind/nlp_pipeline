# Custom Model Summary

## Objective

The objective was to improve static vocabulary-knowledge estimation with a deterministic fixed-200-word query protocol and Balanced Accuracy as the primary metric. The target was `Balanced Accuracy >= 0.95` by `q <= 200`.

That target was not reached. The best confirmed full 16-user fastText result is:

| Model | Query sequence | BA@50 | BA@100 | BA@200 | Accuracy@200 | NLL@200 | Brier@200 | AUROC@200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `neural_n2c_rate35_svd_r5_hybrid_w30_70` | user-discriminative fixed 200 | 0.802799 | 0.805994 | 0.810475 | 0.792643 | 0.459026 | 0.148451 | 0.886989 |

No evaluated model reached `BA >= 0.90`, and none approached `BA >= 0.95`.

Current detailed artifacts are in `reports/model_improvement_fasttext/`:

| Artifact | Contents |
|---|---|
| `leaderboard_fasttext_static.csv` | Main fastText leaderboard across all current variants. |
| `leaderboard_fasttext_static.md` | Markdown copy of the leaderboard. |
| `hybrid_w30_70_userdisc_all.csv` | Best model per-user LOOU results. |
| `hybrid_w70_30_userdisc_all.csv` | Neural-heavy hybrid weight check. |
| `neural_n2c_rate35_e8_userdisc_all.csv` | Longer/larger neural retraining run. |
| `user_knn_k5_centered_userdisc_all.csv` | Centered user-response KNN diagnostic run. |
| `svd_ridge_r5_centered_userdisc_all.csv` | Centered SVD diagnostic run. |

## Dataset And Processing

The benchmark uses the repository's processed static vocabulary dataset under `data/processed/`.

| Asset | Path | Role |
|---|---|---|
| Words | `data/processed/words.csv` | Master word table. |
| Word frequencies | `data/processed/frequency.csv` | Optional frequency metadata, especially `log_frequency`. |
| Word embeddings | `data/processed/embeddings.npy` | Current 300-dimensional fastText vectors. |
| Static responses | `data/processed/responses_static.csv` | Binary user-word labels. |
| Splits | `data/splits/*.json` | Existing LOOU, validation-user, and cold-word metadata. |

Current processed data:

| Quantity | Value |
|---|---:|
| Words | 31,276 |
| Embedding backend | `fasttext` |
| fastText source | `data/raw/wiki-news-300d-1M.vec.zip` |
| Raw embedding shape | `(31276, 300)` |
| Final word-feature shape | `(31276, 302)` |
| Static response rows | 191,952 |
| Users | 16 |
| Labels per user | 11,997 |
| Static-vocabulary fastText OOV rows | 2 / 11,997 |
| Total OOV rows across all words | 8,073 / 31,276 |

The current embedding is the lighter official fastText `wiki-news-300d-1M.vec.zip` file. It is not the larger `cc.en.300.bin` subword model. Static coverage is effectively complete, so fastText changed the semantic feature basis but did not materially change the ceiling of the evaluated models.

Final word features are built as:

```text
[300-d fastText embedding, word_length, log_frequency]
```

Processing details:

| Step | Description |
|---|---|
| Word indexing | `word_id` values are mapped to contiguous `word_idx` values. |
| User indexing | `user_id` values are mapped to contiguous `user_idx` values. |
| Scalar features | Missing length and frequency values are filled with `0.0`. |
| Embedding alignment | `embeddings.npy` rows are aligned to `words.csv` row order. |
| Label frame | Responses are converted to `user_idx`, `word_idx`, `label`, `user_id`, `word_id`. |
| Static-only scope | Temporal responses are not used in this improvement pass. |

## Train, Validation, And Test Splits

The main reported evaluation is leave-one-user-out over all 16 users.

For each split:

| Phase | Data used |
|---|---|
| Fit | All rows from the other 15 users. |
| Online update | The held-out user's first `q` queried fixed-sequence labels. |
| Evaluation | The held-out user's remaining labeled words not used as queries. |

Budgets reported are `q = 50, 100, 200`.

The repository also contains validation/test user metadata in `data/splits/static_validation_users.json`:

```text
validation_users = [user14, user9, user5]
test_users = [user4, user3, user11, user0, user15, user6, user13, user10, user12, user8, user1, user7, user2]
```

Early neural architecture iteration used validation-user sweeps. The final numbers in this file are full 16-user LOOU averages after the promising candidates were selected. Because hyperparameters were iterated repeatedly on this small dataset, these scores should be treated as experimental model-development results, not a locked final blind-test benchmark.

## Query Protocols

Two deterministic fixed-200 query sequences were evaluated.

| Sequence | Description | Outcome |
|---|---|---|
| `difficulty` | Existing difficulty-stratified and embedding-diverse sequence. | Stable baseline protocol. |
| `user_discriminative` | New supervised sequence selecting words with high cross-user variance, coverage, entropy, and low redundancy in user-response pattern. | Best protocol for personalization, improving BA@200 by roughly 0.006-0.008 for the best variants. |

Both sequences use `seed=42`. The same global 200-word sequence is shared across users and models. For a held-out user, only fixed-sequence words with available labels are revealed; metrics are computed only on unqueried labeled words.

## Implemented Model Families

### Encoder-Decoder RNN Models

Implemented in `src/vocab_benchmark/estimators/neural.py`.

Architectures:

| Variant | Encoder | Decoder |
|---|---|---|
| `gru_mlp` | GRU user-state encoder | MLP decoder over user state and word features. |
| `lstm_bilinear` | LSTM user-state encoder | Bilinear user-state/word decoder. |
| `residual_gru_gated` | Residual GRU encoder | Feature-gated decoder. |

Fitting strategies implemented:

| Strategy | Purpose |
|---|---|
| Teacher-forced sequence training | Train from random observed prefixes. |
| Curriculum prefix training | Move from shorter to longer observation prefixes. |
| Contrastive hard-negative sampling | Improve local decision boundaries around similar words. |

These models implement the common estimator interface: `fit`, `initialize_user_state`, `update_user_state`, `predict_proba`, and `predict_uncertainty`.

### Advanced Neural Memory/MIRT Model

Implemented in `src/vocab_benchmark/estimators/neural_advanced.py`.

This is the strongest pure-neural family. The best quick-fit neural configuration is `n2c_rate35`.

Core structure:

| Component | Technical description |
|---|---|
| Input features | 302-d vector: fastText embedding, word length, log frequency. |
| Online ability state | `theta` with shape `(1, ability_dim)`. |
| Observation encoder | Concatenates word features with binary label, projects through `Linear(feature_dim + 1, hidden_dim)`, `tanh`, dropout. |
| Recurrent update | `GRUCell(hidden_dim, ability_dim)` updates `theta` over observed labels. |
| Discrimination net | `Linear(feature_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, ability_dim)`. |
| Bias net | `Linear(feature_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)`. |
| Memory keys | `Linear(feature_dim + 1, hidden_dim)` for observed word-label memories. |
| Memory queries | `Linear(feature_dim, hidden_dim)` for candidate words. |
| Memory residual | Attention-like residual from observed memories, scaled by a learned scalar. |
| Prior logit | Smoothed train-word prior logit added to candidate score. |
| User-rate bias | Online observed-label-rate bias added after prefix observation. |
| Dynamic centering | Candidate logits are median-centered for balanced-accuracy optimization. |

Best quick-fit neural hyperparameters:

| Parameter | Value |
|---|---:|
| `hidden_dim` | 64 |
| `ability_dim` | 32 |
| `lr` | 0.001 |
| `dropout` | 0.1 |
| `max_epochs` | 2 |
| `weight_decay` | 0.0001 |
| `hard_negative_k` | 4 |
| `target_batch_size` | 96 |
| `min_prefix_len` | 8 |
| `max_prefix_len` | 64 |
| `prior_logit_weight` | 0.8 |
| `user_rate_weight` | 1.0 |
| `dynamic_centering_weight` | 1.0 |
| `user_rate_centering_weight` | 0.35 |

Longer/larger retraining was also run with fastText:

| Parameter | Value |
|---|---:|
| `hidden_dim` | 128 |
| `ability_dim` | 64 |
| `lr` | 0.0003 |
| `dropout` | 0.2 |
| `max_epochs` | 8 |
| `early_stopping_patience` | 2 |
| `hard_negative_k` | 6 |
| `target_batch_size` | 128 |
| `min_prefix_len` | 16 |
| `max_prefix_len` | 128 |

The longer/larger model underperformed the smaller quick-fit model: `BA@200=0.801388` versus `0.805040` for the quick-fit neural user-discriminative run. The likely cause is the very small number of train users per LOOU split; increasing model capacity mostly increases variance rather than adding reliable information.

### SVD Ridge User Estimator

Implemented in `src/vocab_benchmark/estimators/svd.py`.

Structure:

| Step | Description |
|---|---|
| Train matrix | Build train-user by word label matrix. |
| Word prior | Smoothed per-word prior from train users. |
| Centering | Subtract word prior from observed labels. |
| Factorization | Compute SVD of centered train residual matrix. |
| Item components | Store `Vt.T * singular_values` for the selected rank. |
| Online fit | Solve closed-form ridge regression from observed held-out labels to user coefficients plus intercept. |
| Prediction | `prior + residual_scale * item_component @ user_coef + intercept`, clipped to probability bounds. |

Best SVD settings:

| Parameter | Value |
|---|---:|
| `rank` | 5 |
| `ridge` | 1.0 |
| `residual_scale` | 0.5 |
| `intercept_ridge` | 1.0 |

This model is simple and competitive because it directly models cross-user response covariance. It reached `BA@200=0.808605` under the user-discriminative query sequence.

### User-Response KNN Estimator

Implemented in `src/vocab_benchmark/estimators/user_knn.py`.

Structure:

| Step | Description |
|---|---|
| Train matrix | Build train-user by word label matrix. |
| Similarity | Compare held-out observed labels against train-user labels on the same query words. |
| Centering | Center both held-out and train labels by word prior, not by `0.5`. |
| Neighbor weighting | Select top-k train users and softmax similarities by temperature. |
| Prediction | Blend word prior with weighted neighbor labels. |

Best tried KNN settings:

| Parameter | Value |
|---|---:|
| `n_neighbors` | 5 |
| `prior_blend` | 0.35 |
| `similarity_temperature` | 0.35 |

Raw user KNN had reasonable AUROC but poor balanced accuracy. A batch-median centering wrapper improved BA, but not enough to beat SVD or the neural/SVD hybrid.

### Calibration And Centering Wrappers

Implemented in `src/vocab_benchmark/estimators/calibrated.py`.

| Wrapper | Description | Outcome |
|---|---|---|
| `OnlineThresholdCalibratedEstimator` | Fits a per-user decision threshold from observed query labels. | Hurt SVD BA; likely overfits the small observed prefix. |
| `BatchMedianCenteredEstimator` | Shifts candidate logits by the batch median to improve balanced thresholding. | Helped KNN, hurt/failed to improve SVD. |

### Best Hybrid

Implemented through `WeightedAveragedEnsembleEstimator`.

Best confirmed model:

```text
0.30 * NeuralMemoryMIRTEstimator(n2c_rate35)
+ 0.70 * SVDRidgeUserEstimator(rank=5, ridge=1.0, residual_scale=0.5)
```

The hybrid works because the neural model contributes nonlinear semantic/rate/memory signals, while SVD contributes direct cross-user response covariance. The SVD-heavy weighting was best at 200 queries.

## FastText Leaderboard

| Run | q | Balanced Accuracy | Accuracy | NLL | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid_w50_50_userdisc` | 50 | 0.804304 | 0.786196 | 0.469544 | 0.152133 | 0.882493 |
| `hybrid_w70_30_userdisc` | 50 | 0.804191 | 0.784051 | 0.477114 | 0.154607 | 0.882927 |
| `hybrid_w30_70_userdisc` | 50 | 0.802799 | 0.788435 | 0.462986 | 0.150092 | 0.882051 |
| `hybrid_w50_50_userdisc` | 100 | 0.806254 | 0.785975 | 0.470623 | 0.152620 | 0.884188 |
| `hybrid_w30_70_userdisc` | 100 | 0.805994 | 0.788424 | 0.463939 | 0.150537 | 0.884478 |
| `hybrid_w70_30_userdisc` | 100 | 0.805749 | 0.783149 | 0.478394 | 0.155186 | 0.884202 |
| `hybrid_w30_70_userdisc` | 200 | 0.810475 | 0.792643 | 0.459026 | 0.148451 | 0.886989 |
| `hybrid_w50_50_userdisc` | 200 | 0.809912 | 0.789544 | 0.466435 | 0.150835 | 0.886589 |
| `hybrid_w70_30_userdisc` | 200 | 0.809000 | 0.785968 | 0.475104 | 0.153778 | 0.886387 |
| `svd_userdisc_best` | 200 | 0.808605 | 0.794821 | 0.453728 | 0.145789 | 0.886451 |
| `neural_n2c_rate35_userdisc` | 200 | 0.805040 | 0.778270 | 0.489089 | 0.158890 | 0.884828 |
| `neural_n2c_rate35_e8_userdisc` | 200 | 0.801388 | 0.775261 | 0.490545 | 0.159708 | 0.880410 |
| `user_knn_k5_centered_userdisc` | 200 | 0.790238 | 0.740464 | 0.553524 | 0.179860 | 0.882038 |

The complete table is in `reports/model_improvement_fasttext/leaderboard_fasttext_static.csv`.

## Why The Target Was Not Reached

The main reason is not embedding quality. The static evaluation vocabulary is almost completely covered by the lighter fastText vectors, yet performance moved only slightly. The bottleneck appears to be information and data geometry.

Key limitations:

| Limitation | Impact |
|---|---|
| Only 16 users total | Each LOOU model trains on 15 users, which is too little to learn high-dimensional personalized neural structure robustly. |
| 200 queries versus about 12k labels per user | The prefix is small relative to the evaluation space. Many user-specific deviations remain unobserved. |
| Strong word-prior dominance | Word difficulty explains much of the signal; personalization adds only modest incremental BA. |
| User idiosyncrasy/noise | Cross-user covariance is not strong enough to infer near-perfect labels from 200 answers. |
| Semantic embeddings weakly tied to individual knowledge | fastText similarity helps word semantics, but knowing semantically similar words is not equivalent to knowing the target word. |
| Balanced accuracy thresholding | Several models have decent AUROC but are hard to threshold per user without overfitting observed labels. |

The strongest evidence is that SVD over train-user response residuals and neural/SVD hybrids all converge near `BA=0.81`, while AUROC is only about `0.887`. A model with AUROC under `0.90` is not realistically going to deliver `BA=0.95` under a fixed threshold without a major new information source.

## Prospects For Further Improvement

Most likely useful next steps:

1. Add many more users. This is the highest-leverage path. The SVD and user-KNN diagnostics show that cross-user response covariance is useful, but 15 train users per split is too few.
2. Replace fixed global queries with adaptive active querying. The fixed-200 protocol is intentionally restrictive. Adaptive query selection targeting the held-out user's uncertainty and latent factors should improve personalization more than another neural architecture pass.
3. Use a richer lexical feature stack. Add CEFR level, word prevalence, morphology, lemma/family features, concreteness, part of speech, frequency by corpus/domain, and multilingual cognate features if relevant.
4. Try subword fastText only if OOV or morphology becomes important. The current static OOV count is only 2 words, so the larger `.bin` model is unlikely to close the gap by itself.
5. Train a hierarchical Bayesian IRT model. A model with word difficulty, user ability, word discrimination, and user clusters may be better calibrated than the current neural model on small user counts.
6. Optimize directly for held-out balanced accuracy using validation users. The current thresholding wrappers were simple. A more principled validation-fitted calibration layer may recover a small amount of BA.
7. Use temporal or repeated observations if available. If the same user has longitudinal data, a model of learning/forgetting and consistency would provide a stronger user signal than static labels alone.
8. Estimate an empirical ceiling. Run oracle diagnostics using larger query budgets and train-user nearest-neighbor upper bounds to quantify whether `0.95` is feasible for the dataset at all.

Most likely not useful by itself:

| Idea | Reason |
|---|---|
| More epochs on the same neural model | The 8-epoch/larger-state run underperformed the smaller model. |
| Larger hidden states only | Data scarcity dominates capacity. |
| Larger fastText model only | Static OOV is already almost zero. |
| Online threshold fitting from 50-200 observed labels only | The tried threshold wrapper overfit and reduced BA. |

## Current Conclusion

The best current design is the SVD-heavy neural hybrid under the user-discriminative fixed-200 sequence. It is reproducible, uses the requested fastText embedding, and outperforms the individual neural and SVD models, but it remains far below the `0.95` balanced-accuracy goal.

To make `BA >= 0.95` feasible, the project likely needs either substantially more users, adaptive querying, richer external lexical/user features, or a different target definition. Continuing to scale the current encoder-decoder architecture on 15 train users per split is unlikely to close the remaining gap.

## Latest Backed-Feature And `q=1000` Accuracy Pass

This pass changed the feature pipeline so that frequency/domain fields are only kept when backed by real data. Unsupported proxy columns were removed.

### Backed Feature Sources

| Source | Implemented fields | Coverage | Notes |
|---|---|---:|---|
| Local L2 workbook: `data/raw/Responses L2 English speakers to 62 thousand words.xlsx`, `Words` sheet | `accuracy`, `acc_L2`, `rank_L2`, `nobs_L2`, `acc_L1`, `rank_L1`, `diff_L1_L2` | 15,194 / 31,276 words | `acc_L2` duplicates the workbook's `accuracy` column so downstream code has an explicit L2 accuracy field. |
| `wordfreq` Python package | `frequency`, `log_frequency`, `wordfreq_zipf` | 31,276 / 31,276 words | Aggregate multi-source Zipf frequency, not a per-domain corpus field. |
| SUBTLEXus / OpenSubtitles-derived list from `words/subtlex-word-frequencies` | `subtlex_us_count`, `log_frequency_subtlex_us`, `subtlex_us_rank`, `subtlex_us_rank_percentile` | 16,560 / 31,276 words | Downloaded to `data/raw/frequency_sources/subtlex_word_frequencies_index.json`; rank and log frequency are derived from real counts. |

Removed fields because no real open/downloaded backing data was incorporated: `log_frequency_commoncrawl`, `log_frequency_academic`, `log_frequency_news`, `log_frequency_spoken`, and the earlier proxy `log_frequency_opensubtitles` column. Domain-specific COCA/iWeb lists exist commercially, but were not added because no local licensed data file was available.

Relevant external sources checked:

| Source | Use |
|---|---|
| https://github.com/words/subtlex-word-frequencies | Open SUBTLEXus count list used in the pipeline. |
| https://pypi.org/project/wordfreq/ | Aggregate Zipf frequency source used for `wordfreq_zipf`. |
| https://www.english-corpora.org/resources.asp | Confirms COCA/iWeb word-frequency/domain lists exist, but as download resources not currently present in this repo. |

Current `data/processed/frequency.csv` has 20 columns and no unsupported proxy frequency fields.

### Feature Sets Re-tested

| Feature set | Shape | Contents |
|---|---:|---|
| `legacy` | `(31276, 302)` | fastText 300d + raw length + `log_frequency`. |
| `fasttext_only` | `(31276, 300)` | fastText 300d only. |
| `l2` | `(31276, 309)` | fastText 300d + normalized length/log frequency + L2/L1 workbook stats. |
| `freq` | `(31276, 313)` | fastText 300d + normalized length + `wordfreq_zipf`, SUBTLEX count/log/rank features, rank percentile, frequency-band one-hot. |
| `rich` | `(31276, 320)` | Combined `freq` + L2/L1 workbook stats. |

### Practical Model Filtering And RAM Use

The standalone `online_user_logreg` model was discarded from further active sweeps: it reached only `accuracy@1000=0.750676`, `BA@1000=0.677741`, and was not competitive. Benchmarking was then restricted to practical models with mean per-user runtimes around one second.

Parallel benchmarking used two worker processes after checking RAM. Four workers were avoided for heavier grids. During two-worker runs, available RAM stayed above roughly 3.4 GiB on a 15 GiB machine and returned to about 5-10 GiB after subprocesses exited. No swap was configured.

### Latest Results At `q=100` And `q=1000`

Primary metric in this phase was raw `accuracy@1000`; `accuracy@100` and balanced accuracy were still tracked.

Top raw-accuracy models at `q=1000`:

| Model | Feature set | Accuracy@100 | BA@100 | Accuracy@1000 | BA@1000 | NLL@1000 | Brier@1000 | AUROC@1000 | Runtime/user |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rasch_twopl_voteuser_grid_w35_45_20` | `rich` | 0.815657 | 0.789666 | 0.840741 | 0.791581 | 0.383573 | 0.118278 | 0.900561 | 0.93s |
| `rasch_twopl_voteuser_grid_w30_50_20` | `rich` | 0.815316 | 0.789884 | 0.840718 | 0.792062 | 0.383438 | 0.118245 | 0.900557 | 0.93s |
| `rasch_twopl_voteuser_grid_w35_50_15` | `rich` | 0.815757 | 0.787498 | 0.840707 | 0.791546 | 0.385006 | 0.118291 | 0.900410 | 0.93s |
| `rasch_twopl_voteuser_grid_w40_40_20` | `rich` | 0.816204 | 0.789359 | 0.840695 | 0.791203 | 0.383714 | 0.118312 | 0.900563 | 0.94s |
| Pure `rasch_highbudget` | `l2`/`rich` irrelevant | 0.817139 | 0.785974 | 0.839644 | 0.776777 | 0.409641 | 0.119110 | 0.896691 | 0.90s |

Top balanced-accuracy models at `q=1000`:

| Model | Feature set | Accuracy@1000 | BA@1000 | Note |
|---|---|---:|---:|---|
| `ft_kernel_smooth` | `rich` | 0.796746 | 0.824246 | Best BA, poor raw accuracy. |
| `ft_kernel` | `l2` | 0.796001 | 0.824185 | Similar BA to smooth kernel. |
| Accuracy-threshold calibrated Rasch/TwoPL/vote-user | `rich` | ~0.8173 | ~0.8076 | Improved BA over raw-focused ensembles but damaged raw accuracy. |

### Interpretation

The new backed features helped, but only modestly. The L2 workbook stats improved the vote-user-logistic residual model from `accuracy@1000=0.831704` (`legacy`) to `0.832682` (`l2`). The frequency-only additions were mostly neutral. Combining frequency and L2 stats (`rich`) was useful in the final ensemble, but the gain over pure Rasch was still only about `+0.0011` absolute raw accuracy.

The best current model is therefore not a larger neural net. It is a lightweight calibrated ensemble:

```text
0.35 * RaschIRTOnlineEstimator
+ 0.45 * TwoPLIRTOnlineEstimator
+ 0.20 * vote_userlogreg_hybrid
```

where `vote_userlogreg_hybrid` is:

```text
0.75 * ObservedMatchUserVoteEstimator(temperature=0.10)
+ 0.25 * OnlineUserLogisticEstimator(C=0.1, prior_blend=0.5, min_observations=50)
```

The ensemble uses the `rich` word feature matrix for the online logistic component. The Rasch and TwoPL components use train-user response statistics and online user ability updates.

### Target Status

The `.95 accuracy@1000` goal was not reached. The best measured result is `accuracy@1000=0.840741`.

The result appears limited by information rather than by compute alone. With only 15 train users in each leave-one-user-out split, the model has strong item-difficulty information but weak evidence about held-out-user-specific idiosyncrasies. The feature additions are mostly global word-level difficulty proxies; they cannot explain much of the remaining per-user disagreement.

Most promising next steps:

1. Add many more train users. This is the highest-leverage path; current collaborative and neural models are starved for user-user covariance.
2. Add licensed COCA/iWeb domain-frequency tables if available locally, especially genre frequencies for academic/news/spoken/fiction/subtitle domains.
3. Train a richer item-response model with side information: item difficulty as a learned function of fastText + L2 stats + frequency, with hierarchical shrinkage instead of direct per-word empirical means.
4. Add per-user latent-topic diagnostics from observed responses, but only if more users are available; with 15 train users, factor models overfit quickly.
5. Treat `.95 raw accuracy` as potentially infeasible on this dataset without either more observed words, more train users, or a different label definition, because the current best models already exploit item difficulty, user ability, and nearest-user residuals.

## Single-Thread Refinement And Cumulative `q=1000` Fix

This pass used one benchmark process and pinned numeric libraries to one thread:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

This was necessary because multi-process benchmarking was unstable under the available RAM and no swap was configured.

### Implementation Correction

A correctness issue was found in the online IRT components. `RaschIRTOnlineEstimator` and `TwoPLIRTOnlineEstimator` were updating from only the newest budget increment. In a run with budgets `100,1000`, the IRT state at `q=1000` therefore did not fully use the first 100 observations. The estimators now store cumulative `observed_word_ids` and `observed_labels` in `UserState` and refit the online ability `theta` against all observations seen so far.

This matters for the current goal because `q=1000` must mean exactly 1000 observed labels, not only the latest increment between budget checkpoints.

### Refined Practical Model

The best current model is a budget-adaptive probability ensemble:

```text
if observed_labels < 500:
    P_known = 0.760 * RaschIRTOnlineEstimator(prior_var=25, steps=20)
            + 0.175 * ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0)
            + 0.065 * OnlineUserLogisticEstimator(C=0.1, prior_blend=0.5, min_observations=50)
    logit_bias = +0.020
else:
    P_known = 0.290 * RaschIRTOnlineEstimator(prior_var=25, steps=20)
            + 0.500 * TwoPLIRTOnlineEstimator(prior_var=25, steps=20)
            + 0.130 * ObservedMatchUserVoteEstimator(temperature=0.10, prior_blend=0.0)
            + 0.080 * OnlineUserLogisticEstimator(C=0.1, prior_blend=0.5, min_observations=50)
    logit_bias = +0.015
```

All components output `P(known | observed held-out-user labels, candidate word)`. The ensemble averages component probabilities, applies the logit bias, clips to `[1e-6, 1 - 1e-6]`, and the benchmark classifies with threshold `0.5`.

The online logistic component uses the `rich` word-feature vector:

```text
300d fastText vector
+ normalized word length
+ wordfreq_zipf / aggregate frequency features
+ backed SUBTLEXus count, log-count, rank percentile
+ frequency-band one-hot features
+ L2 workbook fields: accuracy, acc_L2, rank_L2, nobs_L2, acc_L1, rank_L1, diff_L1_L2
```

### Corrected Benchmark Results

Benchmark protocol:

| Field | Value |
|---|---|
| Dataset | `responses_static` |
| Split | 16-user leave-one-user-out over the selected static users |
| Query sequence | fixed `user_discriminative` sequence |
| Evaluation | unqueried labeled words after each budget |
| Threads | one process, one BLAS/OpenMP thread |
| Artifact | `reports/model_improvement_fasttext/backed_feature_grid/budget_adaptive_refined_raw_rich_summary.csv` |

Results for the refined budget-adaptive model:

| q | Accuracy | Balanced Accuracy | NLL | Brier | AUROC | Mean runtime/user |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.819345 | 0.785181 | 0.419554 | 0.132277 | 0.884917 | 0.464s |
| 1000 | 0.841587 | 0.794700 | 0.382649 | 0.118515 | 0.900734 | 0.937s |

The best direct high-budget candidate was also benchmarked independently:

```text
direct_rasch_twopl_vote_user_grid_290_500_130_080_015
```

It reached the same `accuracy@1000=0.841587`, confirming that the corrected budget-adaptive wrapper now uses all 1000 observations.

A narrow real-evaluator sweep around the best high-budget weights found no better threshold:

| High-budget config | Accuracy@1000 | BA@1000 |
|---|---:|---:|
| `290_500_130_080`, bias `+0.015` | 0.841587 | 0.794700 |
| `315_480_125_080`, bias `+0.015` | 0.841582 | 0.794505 |
| `310_465_145_080`, bias `+0.015` | 0.841576 | 0.794433 |
| `380_400_155_065`, bias `+0.015` | 0.841576 | 0.793576 |
| `290_500_130_080`, bias `+0.020` | 0.841508 | 0.794370 |
| `290_500_130_080`, bias `+0.010` | 0.841417 | 0.794729 |
| `290_500_130_080`, bias `+0.025` | 0.841332 | 0.793935 |

### Target Status

The `0.95 accuracy@1000` target was not reached. The best corrected measured result is:

```text
accuracy@1000 = 0.841587
balanced_accuracy@1000 = 0.794700
```

The latest improvement over the previous best raw `accuracy@1000=0.840741` is real but small: about `+0.00085` absolute. The ceiling appears dominated by data signal rather than by training time. The models already use global item difficulty, online user ability, nearest-user response voting, fastText-backed word features, and L2 workbook difficulty fields. The remaining errors likely come from held-out-user-specific vocabulary idiosyncrasies that are weakly predictable from only 15 train users.

### Concrete Next Steps

1. Increase the number of train users. This is still the highest-leverage path for reaching 0.95 because the best residual models need more user-user covariance.
2. Add true adaptive querying. The fixed sequence is not optimized for each held-out user; an active policy that targets uncertainty over user ability and residual clusters should make the 1000 observations more informative.
3. Fit a hierarchical side-information IRT model. Use fastText, L2 accuracy, and frequency features to predict item difficulty/discrimination with shrinkage, instead of relying mostly on empirical per-word means.
4. Add external lexical resources only when backed by real files: CEFR, concreteness, morphology, POS, word prevalence, and licensed domain-frequency tables if available.
5. Estimate a data ceiling. Run oracle diagnostics that use held-out labels or nearest-user upper bounds to determine whether 0.95 is feasible on the current 16-user static dataset.
6. Avoid broad neural retraining until the data ceiling is clearer. Prior neural variants were slower and did not beat the practical IRT/vote/logistic ensembles.

Validation status: `13 passed in 7.87s`.
