# Grouped Residual IRT: Formal Reimplementation Specification

This document specifies the grouped residual IRT family as implemented in this repository, with full details needed to reimplement it from scratch.

## 1. Problem Setup

We model binary word knowledge:
- Users: \(u \in \{1,\dots,U\}\)
- Words/items: \(i \in \{1,\dots,N\}\)
- Observed label: \(y_{u,i} \in \{0,1\}\), where 1 = known, 0 = unknown.

Goal at inference time for one user \(u\): estimate
\[
P(y_{u,i}=1 \mid \mathcal D_u)
\]
for candidate words \(i\), where \(\mathcal D_u\) is the set of observed quiz answers for that user.

## 2. Model Family in This Repo

There are two grouped residual IRT implementations:

1. `GroupedResidualIRTOnlineEstimator`
- Logistic IRT with user scalar ability \(\theta_u\) and user group residual vector \(\mathbf r_u \in \mathbb R^G\).
- Soft word-group matrix \(\mathbf W \in \mathbb R^{N\times G}\) built from word embeddings.

2. `Response12GroupedResidualIRTEstimator` (current best grouped model used in CLI as `best_grouped_irt_model`)
- Logistic IRT with \(\theta_u\), user residual vector \(\boldsymbol\delta_u \in \mathbb R^G\), observation-count gate \(g(n)\), and per-user threshold calibration.
- Soft group matrix \(\mathbf Q \in \mathbb R^{N\times G}\) built from item response patterns across users (not from embedding clustering).

The current production grouped model is (2), with name:
`response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk`.

---

## 3. Core Prediction Equation (Production Variant)

For user \(u\), word \(i\), and observed count \(n_u = |\mathcal D_u|\):
\[
z_{u,i} = \theta_u - b_i + g(n_u)\,\mathbf q_i^\top \boldsymbol\delta_u,
\]
\[
P(y_{u,i}=1\mid\mathcal D_u)=\sigma(z_{u,i}), \quad \sigma(x)=\frac{1}{1+e^{-x}}.
\]

Gate:
\[
g(n)=\frac{n}{n+c}, \quad c>0.
\]
In the best model: \(c=12\).

Interpretation:
- \(b_i\): global item difficulty (higher \(b_i\) means harder).
- \(\theta_u\): global user ability.
- \(\mathbf q_i\): soft membership of item \(i\) in \(G\) groups.
- \(\boldsymbol\delta_u\): user-specific residual strength per group.
- \(g(n)\): suppresses residual effects at tiny budgets, increases with more observations.

---

## 4. Training/Fit Stage

### 4.1 Inputs
- `train_responses`: table with columns `user_idx`, `word_idx`, `label`.
- `word_features`: only used here for number of words \(N\) in production variant.
- Optional external accuracy prior (from `data/processed/frequency.csv`, column `accuracy`).

### 4.2 Difficulty Construction \(b_i\)

If `use_accuracy_difficulty=True` (best model setting):
1. Load per-word accuracy \(a_i\) from `accuracy` column.
2. Convert percent to fraction if needed: if \(a_i>1\), use \(a_i/100\).
3. Replace NaN by 0.5.
4. Clip: \(p_i=\mathrm{clip}(a_i,10^{-4},1-10^{-4})\).
5. Convert to raw difficulty:
\[
b_i^{\text{raw}} = -\log\frac{p_i}{1-p_i}.
\]
6. Standardize globally using observed words:
\[
b_i = \frac{b_i^{\text{raw}}-\mu_b}{\sigma_b},
\]
where \(\mu_b,\sigma_b\) are mean/std over words seen in `train_responses` (fallback: all words). If \(\sigma_b\le 10^{-8}\), set all \(b_i=0\).

If `use_accuracy_difficulty=False`, prior \(p_i\) is built from empirical mean labels in train responses.

### 4.3 Response Matrix for Group Discovery

Build user-item matrix \(Y\in\mathbb R^{U\times N}\) with NaN for missing labels.

Item-majority imputation:
- For each item \(i\), compute item mean \(\bar y_i\) over observed users.
- Fill missing values in column \(i\) by hard majority:
\[
\tilde y_{u,i} = \mathbb 1[\bar y_i\ge 0.5].
\]
This yields dense binary matrix \(\tilde Y\).

Only words observed at least once are clustered; unobserved words get uniform group weights.

### 4.4 Group Matrix \(\mathbf Q\) (`_build_response12_groups`)

Let \(X\in\mathbb R^{M\times U}\) be transposed observed submatrix (items × users), where \(M\) is number of observed items.

1. Row-center each item response vector:
\[
\mathbf x_i \leftarrow \mathbf x_i - \frac{1}{U}\sum_{j=1}^{U}x_{ij}.
\]
2. L2-normalize each row:
\[
\hat{\mathbf x}_i = \frac{\mathbf x_i}{\max(\|\mathbf x_i\|_2,10^{-8})}.
\]
3. Run KMeans on \(\hat X\) with \(G\) clusters (`n_init=20`, fixed `random_state`).
4. Normalize cluster centers to unit norm.
5. Compute cosine similarity matrix:
\[
S_{ik} = \hat{\mathbf x}_i^\top \hat{\mathbf c}_k.
\]
6. Sparse top-3 soft assignment per item:
- take top 3 clusters by \(S_{ik}\), call this index set \(T_i\).
- local logits: \(\ell_{ik}=6 S_{ik}\) for \(k\in T_i\).
- softmax over \(T_i\).
- force dominant cluster mass to at least 0.5, then renormalize.
- set other clusters to 0.

So each observed item has at most 3 nonzero entries in \(\mathbf q_i\), with dominant mass \(\ge 0.5\).

For items never observed in train responses:
\[
\mathbf q_i = (1/G,\dots,1/G).
\]

### 4.5 How to Regenerate Groupings Exactly

Use this exact protocol to reproduce \(\mathbf Q\) for `Response12GroupedResidualIRTEstimator`:

1. Prepare `train_responses` with columns `user_idx`, `word_idx`, `label` (binary), with `word_idx` aligned to `words.csv`.
2. Build user-item matrix \(Y\) with NaN for missing labels.
3. Compute item-majority imputation as in section 4.3 to get dense \(\tilde Y\).
4. Build `observed_word_ids` as sorted unique `word_idx` present in `train_responses`.
5. Build item-by-user matrix:
\[
X=\tilde Y[:, \text{observed\_word\_ids}]^T.
\]
6. Row-center and L2-normalize each row of \(X\).
7. Run KMeans with exact settings:
   - `n_clusters = G`
   - `random_state = seed`
   - `n_init = 20`
8. L2-normalize cluster centers.
9. Compute cosine similarities \(S\) between normalized item rows and normalized centers.
10. For each item row:
   - keep only top-3 clusters by similarity;
   - logits = `6.0 * similarity`;
   - softmax over those top-3 logits only;
   - force dominant cluster probability to be at least `0.5`;
   - renormalize and set all non-top-3 entries to 0.
11. Initialize full \(N\times G\) matrix with `1/G`, then overwrite rows indexed by `observed_word_ids` with computed soft rows.

Reproducibility constraints:
- Grouping changes if `train_responses` changes (different users/labels/ordering-independent set).
- Grouping changes if `seed` changes.
- In CLI, if `--seed` is omitted, a time-based seed is generated (`time.time_ns() % (2**31 - 1)`), so groupings are not repeatable across runs unless seed is fixed.
- `word_idx` alignment across `responses_static.csv`, `words.csv`, and `frequency.csv` must be exact.

---

## 5. Online User Update (Production Variant)

Given prior user state and newly observed pairs \((i_t,y_t)\):
1. Append observations to persistent history.
2. Let all observed ids/labels be \(\{(i_m,y_m)\}_{m=1}^{n}\).
3. Gate value: \(g=g(n)=n/(n+c)\).
4. Class-balanced sample weights:
   - \(r=\mathrm{clip}(\text{mean}(y),0.05,0.95)\)
   - \(w_+=0.5/r\), \(w_-=0.5/(1-r)\)
   - \(w_m = y_m w_+ + (1-y_m) w_-\)

### 5.1 Initialization of \(\theta\)

Before joint optimization, solve 1D Rasch-only objective:
\[
\min_{\theta\in[-6,6]} \sum_{m=1}^n \Big(y_m\log(1+e^{-(\theta-b_{i_m})}) + (1-y_m)\log(1+e^{\theta-b_{i_m}})\Big)
+\frac{\theta^2}{2\tau_\theta^2}.
\]
Optimized with bounded scalar minimization.

### 5.2 Joint Objective for \((\theta,\boldsymbol\delta)\)

Define for each sample:
\[
z_m = \theta - b_{i_m} + g\,\mathbf q_{i_m}^\top\boldsymbol\delta,
\quad p_m=\sigma(z_m).
\]

Minimize weighted negative log-likelihood + Gaussian priors:
\[
\mathcal L(\theta,\boldsymbol\delta)=
-\sum_{m=1}^n w_m\Big[y_m\log p_m+(1-y_m)\log(1-p_m)\Big]
+\frac{\theta^2}{2\tau_\theta^2}
+\frac{\|\boldsymbol\delta\|_2^2}{2\tau_\delta^2}.
\]

Gradients:
\[
\frac{\partial \mathcal L}{\partial \theta} = \sum_m w_m(p_m-y_m)+\frac{\theta}{\tau_\theta^2},
\]
\[
\frac{\partial \mathcal L}{\partial \boldsymbol\delta} = g\sum_m w_m(p_m-y_m)\mathbf q_{i_m}+\frac{\boldsymbol\delta}{\tau_\delta^2}.
\]

Numerical optimizer:
- L-BFGS-B
- `maxiter=200`, `ftol=1e-9`
- If optimization fails, keep initialization.

### 5.3 Per-user Decision Threshold Optimization

After obtaining \((\theta,\boldsymbol\delta)\), compute predicted probs on observed items and pick threshold maximizing balanced accuracy:
- scan thresholds from `threshold_min` to `threshold_max` with `threshold_step`.
- tie-break by threshold closer to 0.5.
- shrink toward 0.5:
\[
t_u = 0.5 + \frac{n}{n+c_t}(t^*-0.5),
\]
where \(c_t=\) `threshold_shrink_c`.

Best model parameters for threshold search:
- `threshold_min=0.10`
- `threshold_max=0.90`
- `threshold_step=0.005`
- `threshold_shrink_c=30.0`

State variance proxy stored as:
\[
\mathrm{var}_u = \max\left(10^{-6},\frac{\tau_\theta^2}{1+n}\right).
\]

---

## 6. Inference

For candidate set \(\mathcal C\):
\[
\hat p_i = \sigma\!\left(\theta_u - b_i + g(n_u)\,\mathbf q_i^\top\boldsymbol\delta_u\right), \quad i\in\mathcal C.
\]
Clip probabilities to \([10^{-6}, 1-10^{-6}]\).

Uncertainty output used in code:
\[
\mathrm{uncertainty}_i = \hat p_i(1-\hat p_i) + 0.05\,\mathrm{var}_u.
\]

Binary decision for UI/reporting can use user threshold \(t_u\):
\[
\hat y_i = \mathbb 1[\hat p_i \ge t_u].
\]

---

## 7. Default Hyperparameters of Current Best Grouped Model

From `scripts/vocab_book_cli.py` (`best_grouped_irt_model`):
- `estimator.name = "response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk"`
- \(G = 12\)
- \(\tau_\theta = 2.0\)
- \(\tau_\delta = 1.6\)
- \(c=\) `gate_c` \(=12.0\)
- `random_state = seed` (from CLI `--seed`)
- `threshold_min = 0.10`
- `threshold_max = 0.90`
- `threshold_step = 0.005`
- `threshold_shrink_c = 30.0`
- `use_accuracy_difficulty = True`
- `accuracy_values = None` (loaded from `data/processed/frequency.csv::accuracy`)

Exact fixed internal constants used by this model implementation:
- Grouping KMeans `n_init = 20`
- Top-k retained group assignments per item: `k = 3`
- Similarity-to-logit scale for grouping softmax: `6.0`
- User-update optimizer: `L-BFGS-B` with `maxiter = 200`, `ftol = 1e-9`
- Rasch-only \(\theta\) initialization: bounded scalar optimization on \([-6,6]\), `xatol = 1e-5`
- Objective-internal probability clipping: \([10^{-8}, 1-10^{-8}]\)
- Exposed prediction clipping: \([10^{-6}, 1-10^{-6}]\)

### 7.1 Per-user Runtime State (Exact Fields)

State payload fields:
- `theta: float`
- `delta: float[G]`
- `threshold: float`
- `var: float`
- `observed_word_ids: int[]`
- `observed_labels: float[]`

Initialization:
- `theta = 0.0`
- `delta = zeros(G)`
- `threshold = 0.5`
- `var = tau_theta^2`
- empty observation arrays

---

## 8. Alternative GroupedResidualIRTOnlineEstimator (Embedding-Grouped Variant)

This variant is also in repo and may be useful to reproduce historical experiments.

### 8.1 Prediction
\[
z_{u,i}=\theta_u-b_i+\mathbf w_i^\top\mathbf r_u,
\quad p_{u,i}=\sigma(z_{u,i}).
\]

### 8.2 Group Matrix \(\mathbf W\) from Embeddings
Input embeddings \(E\in\mathbb R^{N\times d}\) (`embedding_dim`). Supported strategies:
- `kmeans_euclidean`: softmax over negative squared distances to Euclidean KMeans centers.
- `kmeans_cosine`: normalize embeddings, KMeans in cosine space, softmax over cosine similarity.
- `pca_quantile`: project to PC1, quantile centers, Gaussian-kernel logits.
- `anchor_cosine`: random anchor words as centers, cosine-softmax assignments.

Temperature parameter controls softness.

### 8.3 Fit Stage
- Estimate \(b_i\) from train label mean by logit transform.
- Build \(\mathbf W\) once.
- For each user with enough observations, fit residual coefficients by ridge regression on residual targets:
\[
\mathbf r_u = \arg\min_{\mathbf r}\|\mathbf y_u-\mathbf p^{\text{prior}}_u-\mathbf W_u\mathbf r\|_2^2 + \lambda\|\mathbf r\|_2^2,
\quad \lambda=1/\sigma_r^2.
\]
- Aggregate user residual vectors to empirical prior mean/variance.

### 8.4 Online Update
Joint Newton-style updates over \((\theta_u,\mathbf r_u)\) on logistic likelihood with Gaussian priors; solve linear system using full Hessian each step, for `n_fit_steps` iterations.

---

## 9. Required Inputs/Artifacts to Reproduce Production Variant Exactly

1. Word index aligned across all files.
2. Response table (`user_idx`, `word_idx`, `label`).
3. Per-word `accuracy` prior aligned by `word_idx`.
4. Hyperparameters listed in section 7.
5. Deterministic random seed for KMeans.

Any mismatch in alignment between `word_idx` and `accuracy` will materially change behavior.

---

## 10. Minimal Reimplementation Checklist

1. Build/validate `word_idx` mapping.
2. Compute standardized difficulties \(b_i\) from `accuracy` prior.
3. Build dense user-item matrix with majority imputation.
4. Build sparse top-3 soft group matrix \(\mathbf Q\) from response-pattern KMeans.
5. Implement user state \((\theta,\boldsymbol\delta,t,\text{history})\).
6. On update, optimize weighted objective via L-BFGS-B.
7. Recompute user threshold via BA sweep + shrink.
8. Predict via gated logit equation.

This reproduces the grouped residual IRT used as `best_grouped_irt_model` in this repository.
