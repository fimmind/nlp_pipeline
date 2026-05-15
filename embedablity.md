# Static HTML Embeddability Analysis of Top Models

## Scope and assumptions

- Target: embed inference in a **static HTML file** (no backend calls).
- Dataset scale from current repo state:
  - words: `N = 31,276`
  - train users: `U = 16`
  - embedding dim: `d = 300`
- "Top models" here means models that were either:
  - top on core benchmark artifacts (`leaderboard_fasttext_static.csv`, grouped-IRT reports), or
  - selected as practical best in CLI/time benchmarks.

## Quick verdict

- **Easiest and safest to embed**: `basic_rasch_from_accuracy`, `rasch_highbudget_var25`, `twopl_irt_online_baseline`.
- **Best tradeoff of quality vs browser complexity**: `twopl_irt_online_baseline` and small blended non-neural ensembles.
- **Feasible but heavier**: grouped residual IRT (+TwoPL hybrid), SVD-based models, observed-vote models.
- **Poor fit for single standalone HTML**: neural memory models and fastText-kernel-heavy hybrids.

---

## Feasibility table

| Model family / representative | Recent quality signal | What must be embedded in HTML | Approx payload (float32) | Runtime profile in browser | Embedding effort |
|---|---|---|---:|---|---|
| **Basic Rasch from accuracy** (`basic_rasch_from_accuracy`) | `accuracy@1000 ~0.814` (`basic_rasch_accuracy16_summary`) | difficulty vector `b[N]` from `accuracy` column | ~0.12 MB | Update: tiny Newton on scalar `theta`; Predict all words: `O(N)` | **Very easy** |
| **Rasch IRT** (`rasch_highbudget_var25`) | `accuracy@1000 ~0.808`, `BA@100 ~0.776` | `b[N]` learned from train responses | ~0.12 MB | Same as above | **Very easy** |
| **TwoPL IRT** (`twopl_irt_online_baseline`) | grouped IRT report winner at `q=1000` BA (`~0.8008`) | `a[N]`, `b[N]` | ~0.25 MB | Update: scalar Newton using `a,b`; Predict all words: `O(N)` | **Easy** |
| **SVD Ridge user model** (`svd_ridge_r5_l1_s05` / `svd_userdisc_best`) | strong in static leaderboard (`BA@200 ~0.8086`) | `word_prior[N]`, `item_components[N,r]`, `r=5` | ~0.75 MB | Per predict: solve tiny ridge (`(r+1)x(r+1)`), then `O(N*r)` | **Easy–medium** |
| **Observed user vote** (`observed_match_user_vote_t10`) | used in practical best ensemble | `user_word_matrix[U,N]`, `word_prior[N]` | ~2.0 MB + 0.12 MB | Weight calc `O(U*K)` + predict `O(U*N)` | **Medium** |
| **Practical non-neural ensemble** (`budget_adaptive_refined_raw_switch500`) | current CLI practical best | Parameters of Rasch + TwoPL + vote + user-logreg branch + blend weights | dominated by vote/logreg features | Branch switch + weighted blend; still fast if components are optimized | **Medium–hard** |
| **Grouped residual IRT** (`grouped_residual_irt_*`) | best grouped variants strong at `q=200`; near TwoPL at `q=1000` | `b[N]`, `group_weights[N,G]`, group priors, optional TwoPL blend params | `G=16`: ~2.0 MB weights; `G=64`: ~8.0 MB | Update: solve `(G+1)x(G+1)` each step block; Predict: `O(N*G)` | **Medium** (`G=16`) / **Hard** (`G=64`) |
| **FastText-kernel logistic** (`fasttext_kernel_logistic`) | helpful component in some hybrids | normalized embeddings `Nxd`, scalar features, logistic coeffs | embeddings alone ~36 MB | Per predict uses candidate-observed similarities: heavy (`O(N*K*d)`) | **Hard** |
| **Neural memory / MIRT** (`neural_memory_mirt_n2c_rate35`) | strong family historically; in top hybrid stacks | NN weights + runtime (PyTorch equivalent in browser) + feature tensors | model small-ish, runtime dependency dominates | Requires NN inference runtime and careful optimization | **Very hard** in one-file static HTML |
| **Neural+SVD+fastText hybrid** (`neural_n2c_svd_ftkernel_hybrid_w20_55_25`) | top in `time_est_summary` at higher budgets | union of neural + kernel + svd assets | very large (tens of MB) | heavy; combines hardest components | **Very hard / impractical** |

---

## Why these ratings

## 1) IRT models are the cleanest static-HTML target

- Rasch/TwoPL need only per-word parameter arrays and scalar user state (`theta`, optional variance).
- Browser implementation is straightforward math (`exp`, `log`, vector loops).
- No heavy runtime dependencies.
- State persistence is tiny (a few floats + observed labels if needed).

## 2) SVD is still practical

- Rank-5 SVD requires only a compact item matrix and prior vector.
- Prediction can be made fast with typed arrays and Web Workers.
- No model retraining in browser is required beyond tiny closed-form user solve.

## 3) Grouped residual IRT is feasible if you keep `G` small

- With precomputed `group_weights`, browser only does online update and matrix solve.
- `G=16` is realistic; `G=64` increases both payload and cubic solve cost in user update.
- If single-file size matters, grouped IRT can be compressed aggressively but still larger than TwoPL.

## 4) Vote/logreg/kernel/neural become integration-heavy

- Vote models need train-user response matrix in payload.
- Online user logistic in current implementation retrains a logistic model during prediction; reproducing this robustly in pure JS is non-trivial.
- FastText-kernel models are dominated by embedding payload and similarity computations.
- Neural models require exporting to ONNX/TF.js or full manual reimplementation; this conflicts with “single standalone HTML” simplicity.

---

## Standalone HTML practicality ranking

1. `basic_rasch_from_accuracy`
2. `rasch_highbudget_var25`
3. `twopl_irt_online_baseline`
4. `svd_ridge_r5_l1_s05`
5. `grouped_residual_irt_*` with `G<=16`
6. `budget_adaptive_refined_raw_switch500` (if simplified to non-logreg branch or precomputed components)
7. full vote/logreg variants
8. fasttext-kernel variants
9. neural variants

---

## Recommended embed path

For a production static HTML target, the most realistic staged plan is:

1. Ship **TwoPL** first (best quality/complexity balance).
2. Add **Rasch fallback** and optional **basic Rasch-from-accuracy** mode.
3. Add **SVD rank-5** as optional “accuracy boost” mode.
4. Evaluate whether grouped residual (`G=16`) gives enough gain to justify size/complexity.
5. Keep neural and fastText-kernel hybrids out of single-file mode unless moving to a multi-asset static app with dedicated ML runtime.

---

## Bottom line

- If strict requirement is **single standalone HTML**, the top-quality model that is still clearly practical is **TwoPL** (possibly with a small non-neural ensemble).
- The highest-complexity top models (neural and kernel-heavy hybrids) are technically embeddable, but not realistically maintainable or lightweight in a one-file static deployment.
