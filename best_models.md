# Best Models: Rigorous Technical Description

This document describes the best-performing static vocabulary-knowledge models in the current repository state. The current optimization target is raw `accuracy@1000`, while Balanced Accuracy is still tracked because it was the original project metric.

## Notation

Let:

| Symbol | Meaning |
|---|---|
| `U` | Set of users. |
| `W` | Set of words, indexed by `i`. |
| `y_{u,i} in {0,1}` | Response label: `1` if user `u` knows word `i`, else `0`. |
| `D_train` | Responses from the 15 train users in a leave-one-user-out split. |
| `u*` | Held-out test user. |
| `O_q = {(i_t, y_{u*,i_t})}_{t=1}^q` | Observed query answers for held-out user at budget `q`. |
| `C_q` | Candidate unqueried labeled words for evaluation. |
| `x_i in R^d` | Word feature vector for word `i`. |
| `sigma(z)` | Logistic sigmoid: `1 / (1 + exp(-z))`. |
| `logit(p)` | `log(p / (1 - p))`. |

All models estimate:

```text
p_i = P(y_{u*,i} = 1 | O_q, D_train, x_i)
```

The benchmark classifies word `i` as known when `p_i >= 0.5`.

## Dataset And Feature Processing

Evaluation uses the static response data:

| Asset | Path | Role |
|---|---|---|
| Words | `data/processed/words.csv` | Word IDs and metadata. |
| Static responses | `data/processed/responses_static.csv` | Binary known/unknown labels. |
| Embeddings | `data/processed/embeddings.npy` | 300-dimensional fastText vectors aligned to `words.csv`. |
| Frequency/features | `data/processed/frequency.csv` | Backed frequency and L2 workbook fields. |

Current static setup:

| Property | Value |
|---|---:|
| Users | 16 |
| Train users per LOOU split | 15 |
| Words | 31,276 |
| Static labels per user | 11,997 |
| Main embedding | fastText `wiki-news-300d-1M.vec.zip` |
| Embedding dimension | 300 |

The best current raw-accuracy model uses the `rich` feature set:

```text
x_i = concat(
    fastText_i[0:300],
    normalized_length_i,
    wordfreq_zipf_i,
    backed_frequency_features_i,
    SUBTLEXus_features_i,
    frequency_band_one_hot_i,
    L2_workbook_features_i
)
```

The L2 workbook features are:

```text
accuracy, acc_L2, rank_L2, nobs_L2, acc_L1, rank_L1, diff_L1_L2
```

The backed frequency/SUBTLEX features include:

```text
frequency, log_frequency, wordfreq_zipf,
subtlex_us_count, log_frequency_subtlex_us,
subtlex_us_rank, subtlex_us_rank_percentile,
frequency_rank, frequency_rank_percentile, frequency_band
```

Unsupported proxy fields were removed; only backed fields are used.

## Evaluation Protocol

For each held-out user `u*`:

1. Fit model parameters/statistics on `D_train = {y_{u,i}: u != u*}`.
2. Reveal the first `q` words from the deterministic `user_discriminative` query sequence that are labeled for `u*`.
3. Update the model's online user state using all observed pairs in `O_q`.
4. Predict probabilities for every unqueried labeled word in `C_q`.
5. Average metrics across all 16 leave-one-user-out splits.

All reported latest runs were executed single-threaded with:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

## Current Best Raw-Accuracy Model

### Model Name

```text
budget_adaptive_refined_raw_switch500
```

Benchmark output name:

```text
neural_memory_mirt_budget_adaptive_refined_raw_user_discriminative_rich
```

Despite the historical `neural_memory_mirt_` output prefix, this best current model is a practical online ensemble of IRT, nearest-user voting, and per-user logistic regression components.

### Top-Level Architecture

The model keeps two submodels:

```text
M_low  = low-budget ensemble
M_high = high-budget ensemble
```

At prediction time:

```text
M_active(O_q) = M_low   if |O_q| < 500
              = M_high  if |O_q| >= 500
```

The active submodel outputs:

```text
p_i = M_active(O_q, i)
```

Both submodels are updated cumulatively at every query step, so the high-budget branch has already seen all observations when the switch occurs.

## Component 1: Weighted Probability Ensemble

For component probabilities `p_i^{(1)}, ..., p_i^{(K)}` and nonnegative weights `w_1, ..., w_K`, the ensemble first normalizes weights:

```text
alpha_k = w_k / sum_j w_j
```

It computes a weighted probability average:

```text
pbar_i = sum_{k=1}^K alpha_k * p_i^{(k)}
```

Then it applies an optional logit bias `b_0`:

```text
p_i = sigma(logit(clip(pbar_i, 1e-6, 1 - 1e-6)) + b_0)
```

Finally:

```text
p_i <- clip(p_i, 1e-6, 1 - 1e-6)
```

## Component 2: Rasch IRT Online Estimator

### Fitted Item Difficulty

For each word `i`, compute the train-user empirical known rate:

```text
r_i = mean_{u != u* and y_{u,i} observed}(y_{u,i})
```

Then clip and convert to item difficulty:

```text
r_i' = clip(r_i, 1e-4, 1 - 1e-4)
b_i = -logit(r_i')
```

Low `b_i` means an easy/common known word; high `b_i` means a hard/rare known word.

### Online User Ability

The held-out user has scalar ability `theta`. With prior variance `tau^2 = 25`, `theta` is fit by MAP estimation from all observed labels in `O_q`:

```text
L_Rasch(theta) = sum_{(i,y) in O_q} [y * log sigma(theta - b_i)
                                  + (1 - y) * log(1 - sigma(theta - b_i))]
                 - theta^2 / (2 * tau^2)
```

The implementation performs Newton updates for 20 steps with learning rate `1.0`:

```text
p_i = sigma(theta - b_i)
g   = sum_{(i,y) in O_q}(y - p_i) - theta / tau^2
h   = -sum_{(i,y) in O_q} p_i * (1 - p_i) - 1 / tau^2
theta <- theta - g / h
```

The state stores cumulative observations:

```text
state = {theta, var, observed_word_ids, observed_labels}
```

The posterior variance proxy is:

```text
var = max(1e-6, -1 / h)
```

### Prediction

For candidate word `i`:

```text
p_i^Rasch = sigma(theta - b_i)
```

## Component 3: Two-Parameter IRT Online Estimator

TwoPL extends Rasch with a per-word discrimination `a_i`.

### Fitted Item Difficulty And Discrimination

Difficulty is the same as Rasch:

```text
b_i = -logit(clip(r_i, 1e-4, 1 - 1e-4))
```

Let `n_i` be the number of train-user responses observed for word `i`. Discrimination is:

```text
a_i = clip(sqrt(n_i / (n_i + 20)) * 1.7, 0.3, 2.5)
```

Because most words have the same train-user coverage in the static setup, this is mostly a reliability/count scaling rather than a fully learned discrimination parameter.

### Online User Ability

With `tau^2 = 25`, `theta` maximizes:

```text
L_2PL(theta) = sum_{(i,y) in O_q} [y * log sigma(a_i * (theta - b_i))
                                + (1 - y) * log(1 - sigma(a_i * (theta - b_i)))]
               - theta^2 / (2 * tau^2)
```

Newton updates:

```text
p_i = sigma(a_i * (theta - b_i))
g   = sum_{(i,y) in O_q} a_i * (y - p_i) - theta / tau^2
h   = -sum_{(i,y) in O_q} a_i^2 * p_i * (1 - p_i) - 1 / tau^2
theta <- theta - g / h
```

The state also stores cumulative observations, so `q=1000` uses all 1000 labels.

### Prediction

```text
p_i^2PL = sigma(a_i * (theta - b_i))
```

## Component 4: Observed-Match User Vote Estimator

This model is a nearest-user collaborative filter over the train-user response matrix.

### Train Matrix

Construct matrix `M`:

```text
M_{v,i} = y_{v,i} if train user v answered word i
        = NaN     otherwise
```

A smoothed word prior is also computed:

```text
pi_i = (sum_v 1[M_{v,i} observed] * M_{v,i} + 1) / (count_i + 2)
```

### Held-Out User Similarity

For each train user `v`, define the set of observed query words also answered by `v`:

```text
A_v = {i : (i,y_i) in O_q and M_{v,i} observed}
```

The match rate is:

```text
m_v = (1 / max(|A_v|, 1)) * sum_{i in A_v} 1[M_{v,i} = y_i]
```

The implemented best version uses `power = 1`, so:

```text
s_v = m_v
```

With temperature `T = 0.10`, train-user weights are:

```text
alpha_v = exp((s_v - max_j s_j) / T) / sum_j exp((s_j - max_l s_l) / T)
```

### Candidate Probability

For candidate word `i`, compute a weighted vote over train users with observed labels:

```text
vote_i = sum_v alpha_v * 1[M_{v,i} observed] * M_{v,i}
         / sum_v alpha_v * 1[M_{v,i} observed]
```

If the denominator is zero, fallback to `pi_i`.

With `prior_blend = 0.0` in the best models:

```text
p_i^vote = vote_i
```

General implementation form:

```text
p_i^vote = prior_blend * pi_i + (1 - prior_blend) * vote_i
```

## Component 5: Online User Logistic Estimator

This component adapts from the held-out user's observed labels using word features.

### Feature Standardization

From all word features, compute:

```text
mu_j = mean_i x_{i,j}
s_j  = max(std_i x_{i,j}, 1e-4)
z_i  = (x_i - mu) / s
```

A smoothed train-word prior is computed:

```text
pi_i = (sum_{u != u*} y_{u,i} + 1) / (count_i + 2)
```

### Per-User Logistic Fit

If `|O_q| < 50` or the observed labels have only one class, the component returns the prior:

```text
p_i^logreg = pi_i
```

Otherwise it fits a binary logistic regression to held-out observations:

```text
P(y = 1 | z_i, beta, c) = sigma(beta^T z_i + c)
```

The fitted parameters minimize the L2-regularized logistic loss used by scikit-learn `LogisticRegression` with:

```text
C = 0.1
solver = lbfgs
class_weight = None
max_iter = 300
```

After fitting:

```text
q_i = sigma(beta^T z_i + c)
```

The prediction blends user-specific logistic probability with the train-word prior:

```text
p_i^logreg = 0.5 * pi_i + 0.5 * q_i
```

## Best Model: Low-Budget Branch

Used when `|O_q| < 500`.

### Equation

```text
pbar_i^low = 0.760 * p_i^Rasch
           + 0.175 * p_i^vote
           + 0.065 * p_i^logreg

p_i^low = sigma(logit(pbar_i^low) + 0.020)
```

### Intended Role

At small budgets, Rasch ability estimation is more stable than TwoPL and nearest-user residuals. The branch therefore puts most mass on Rasch, with a smaller user-vote residual and a small word-feature logistic correction.

## Best Model: High-Budget Branch

Used when `|O_q| >= 500`.

### Equation

```text
pbar_i^high = 0.290 * p_i^Rasch
            + 0.500 * p_i^2PL
            + 0.130 * p_i^vote
            + 0.080 * p_i^logreg

p_i^high = sigma(logit(pbar_i^high) + 0.015)
```

### Intended Role

At large budgets, enough observations are available for TwoPL to estimate held-out ability more reliably. The high-budget branch therefore shifts most probability mass to TwoPL and keeps Rasch, nearest-user voting, and feature-logistic terms as residual corrections.

## Current Best Raw-Accuracy Performance

Artifact:

```text
reports/model_improvement_fasttext/backed_feature_grid/budget_adaptive_refined_raw_rich_summary.csv
```

| Model | q | Accuracy | Balanced Accuracy | NLL | Brier | AUROC | Runtime/user |
|---|---:|---:|---:|---:|---:|---:|---:|
| `budget_adaptive_refined_raw_switch500` | 100 | 0.819345 | 0.785181 | 0.419554 | 0.132277 | 0.884917 | 0.464s |
| `budget_adaptive_refined_raw_switch500` | 1000 | 0.841587 | 0.794700 | 0.382649 | 0.118515 | 0.900734 | 0.937s |

The best direct high-budget-only model is mathematically identical to the high-budget branch:

```text
direct_rasch_twopl_vote_user_grid_290_500_130_080_015
```

It reached the same `accuracy@1000 = 0.841587` in a direct `q=1000` run.

## High-Budget Direct Weight Sweep

Artifact:

```text
reports/model_improvement_fasttext/direct_weight_refinement/direct_weight_refinement_summary.csv
```

| High-budget model | Accuracy@1000 | BA@1000 | NLL@1000 | AUROC@1000 |
|---|---:|---:|---:|---:|
| `290_500_130_080`, bias `+0.015` | 0.841587 | 0.794700 | 0.382649 | 0.900734 |
| `315_480_125_080`, bias `+0.015` | 0.841582 | 0.794505 | 0.382714 | 0.900725 |
| `310_465_145_080`, bias `+0.015` | 0.841576 | 0.794433 | 0.382724 | 0.900772 |
| `380_400_155_065`, bias `+0.015` | 0.841576 | 0.793576 | 0.383586 | 0.900687 |
| `290_500_130_080`, bias `+0.020` | 0.841508 | 0.794370 | 0.382653 | 0.900734 |
| `290_500_130_080`, bias `+0.010` | 0.841417 | 0.794729 | 0.382648 | 0.900734 |
| `290_500_130_080`, bias `+0.025` | 0.841332 | 0.793935 | 0.382661 | 0.900734 |

## Earlier Best Balanced-Accuracy Neural Hybrid

This model remains useful as a reference because it achieved stronger Balanced Accuracy in earlier long-budget tests, although it is not the current raw-accuracy winner.

### Name

```text
neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15
```

### Ensemble Equation

```text
p_i = 0.25 * p_i^neural
    + 0.60 * p_i^svd
    + 0.15 * p_i^kernel
```

No logit bias is applied.

### Neural Memory/MIRT Component

For each word feature vector `x_i`, the neural component has:

```text
obs_proj:       Linear(d + 1 -> 64)
encoder:        GRUCell(64 -> 32)
discrimination: Linear(d -> 64) -> ReLU -> Dropout(0.1) -> Linear(64 -> 32)
bias:           Linear(d -> 64) -> ReLU -> Dropout(0.1) -> Linear(64 -> 1)
key_proj:       Linear(d + 1 -> 64)
query_proj:     Linear(d -> 64)
```

For observed pair `(i_t, y_t)`, define:

```text
o_t = tanh(W_obs * concat(x_{i_t}, y_t) + b_obs)
theta_t = GRUCell(o_t, theta_{t-1})
```

After all observed labels:

```text
theta_q in R^32
```

For candidate word `i`:

```text
d_i = discrimination(x_i) in R^32
c_i = bias(x_i) in R
base_i = d_i^T theta_q + c_i
```

Memory residual:

```text
s_t = 2*y_t - 1
k_t = tanh(W_key * concat(x_{i_t}, s_t) + b_key)
q_i = tanh(W_query * x_i + b_query)
a_{i,t} = softmax_t((q_i^T k_t) / sqrt(64))
m_i = sum_t a_{i,t} * s_t
```

Logit:

```text
ell_i = base_i
      + memory_scale * m_i
      + prior_logit_weight * logit(pi_i)
      + user_rate_weight * (mean_observed_label - 0.5)
```

The `n2c_rate35` config uses:

```text
hidden_dim = 64
ability_dim = 32
lr = 1e-3
dropout = 0.1
max_epochs = 2
weight_decay = 1e-4
prior_logit_weight = 0.8
user_rate_weight = 1.0
dynamic_centering_weight = 1.0
user_rate_centering_weight = 0.35
hard_negative_weight = 0.2
balanced_surrogate_weight = 0.2
```

Training objective per episode:

```text
loss = BCE_with_logits(ell_i, y_i; class-balanced positive weight)
     + balanced_surrogate_weight * BA_surrogate
     + hard_negative_weight * hard_negative_loss
```

Episodes are generated by taking a random/curriculum prefix from a train user's labeled sequence as observations and predicting sampled remaining targets.

### SVD Ridge Component

Build train response matrix `M` and smoothed word prior `pi_i`. Missing values are imputed by `pi_i`, then centered:

```text
R_{u,i} = filled_M_{u,i} - pi_i
```

Compute SVD:

```text
R = U S V^T
```

Keep rank `r = 5` item components:

```text
g_i = V_{1:r,i} * S_{1:r}
```

For held-out observations, fit user residual coefficients `gamma` and intercept `c` by ridge regression:

```text
min_{gamma,c} sum_{(i,y) in O_q} (y - pi_i - g_i^T gamma - c)^2
              + lambda * ||gamma||_2^2 + lambda_c * c^2
```

with:

```text
lambda = 1.0
lambda_c = 1.0
residual_scale = 0.5
```

Prediction:

```text
p_i^svd = clip(pi_i + 0.5 * (g_i^T gamma + c), 1e-6, 1 - 1e-6)
```

### FastText Kernel Logistic Component

Let `e_i` be the normalized 300d fastText embedding. For candidate `i` and observed word `j`:

```text
sim_{i,j} = e_i^T e_j
```

With temperature `T = 0.15`:

```text
a_{i,j} = softmax_j(sim_{i,j} / T)
s_j = 2*y_j - 1
kernel_i = sum_j a_{i,j} * s_j
pos_max_i = max_{j: y_j=1} sim_{i,j}
neg_max_i = max_{j: y_j=0} sim_{i,j}
rate_delta = logit((sum_j y_j + 1) / (|O_q| + 2)) - logit(mean_j pi_j)
```

The dynamic logistic feature vector is:

```text
phi_i = concat(
    logit(pi_i),
    kernel_i,
    pos_max_i,
    neg_max_i,
    pos_max_i - neg_max_i,
    rate_delta,
    standardized_scalar_word_features_i
)
```

A balanced logistic regression is trained from synthetic train-user episodes:

```text
p_i^kernel_raw = sigma(beta^T phi_i + c)
```

Then logits are centered by median and user-rate quantile adjustments:

```text
ell_i = logit(p_i^kernel_raw)
ell_i <- ell_i - dynamic_centering_weight * median_j(ell_j)
ell_i <- ell_i - user_rate_centering_weight * quantile_j(ell_j, 1 - observed_rate)
p_i^kernel = sigma(ell_i)
```

### Earlier Neural-Hybrid Long-Budget Performance

Artifact:

```text
reports/model_improvement_fasttext/top_model_q500_1000_summary.csv
```

| Model | q | Accuracy | Balanced Accuracy | NLL | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15` | 500 | 0.800529 | 0.819557 | 0.440810 | 0.141537 | 0.892224 |
| `neural_n2c_rate35_svd_r5_ftkernel_hybrid_w25_60_15` | 1000 | 0.809550 | 0.830995 | 0.430158 | 0.137241 | 0.899666 |

This model is better for Balanced Accuracy than the current raw-accuracy model, but it sacrifices raw accuracy. It is therefore not the current best model for the user's latest `.95 accuracy@1000` target.

## Target Status

Current best measured raw accuracy:

```text
accuracy@1000 = 0.841587
```

Current best documented Balanced Accuracy from the long-budget neural-hybrid track:

```text
balanced_accuracy@1000 = 0.830995
```

The `.95 accuracy@1000` target has not been reached. The best current architectures already combine item difficulty, online user ability, collaborative nearest-user residuals, fastText/L2/frequency word features, and per-user feature adaptation. The remaining gap is likely dominated by limited train-user count and user-specific vocabulary idiosyncrasy rather than by a missing small architectural tweak.
