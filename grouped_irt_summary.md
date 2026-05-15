# Grouped Residual IRT: Recent Results

## What was implemented
- Added `GroupedResidualIRTOnlineEstimator` in `src/vocab_benchmark/estimators/irt.py`.
- Added soft grouping strategies derived from fastText embeddings:
  - `anchor_cosine`
  - `pca_quantile`
  - `kmeans_cosine`
  - `kmeans_euclidean`
- Added online joint update of user ability (`theta`) + group residual vector.
- Added numerical-stability fallbacks for linear solves/inversion.
- Added experiment runner: `scripts/run_grouped_residual_irt_experiments.py`.
- Added optional grouped+TwoPL ensemble sweep and logit-bias sweep support.

## Main sweep results (q = 100, 200, 1000)
Source: `reports/model_improvement_fasttext/grouped_residual_irt_focus_round2/focus_round2_best_by_budget.csv`

| q | Best overall estimator | Best overall BA | Best grouped-only BA | Best grouped+TwoPL BA | TwoPL baseline BA |
|---:|---|---:|---:|---:|---:|
| 100 | `grouped_residual_irt_anchor_cosine_g16_t010_rp200_twopl_hybrid_w60_40` | 0.787402 | 0.786454 | 0.787402 | 0.787138 |
| 200 | `grouped_residual_irt_pca_quantile_g8_t025_rp100_twopl_hybrid_w60_40` | 0.792237 | 0.791277 | 0.792237 | 0.789221 |
| 1000 | `twopl_irt_online_baseline` | 0.800754 | 0.786130 | 0.795199 | 0.800754 |

Key point:
- Grouped residual IRT (+ TwoPL ensemble) improves Balanced Accuracy at `q=200`.
- At `q=1000`, TwoPL baseline remains best in Balanced Accuracy.

## Targeted q=1000 bias-calibration pass
Source: `reports/model_improvement_fasttext/grouped_residual_irt_q1000_bias_final/q1000_bias_best_summary.csv`

| Category | Estimator | BA | Accuracy | AUROC | Delta vs TwoPL BA |
|---|---|---:|---:|---:|---:|
| Best grouped-only | `grouped_residual_irt_anchor_cosine_g64_t010_rp100` | 0.786130 | 0.840087 | 0.899395 | -0.014624 |
| Best grouped+TwoPL | `grouped_residual_irt_anchor_cosine_g64_t010_rp100_twopl_hybrid_w50_50_bm060` | 0.797931 | 0.839615 | 0.899394 | -0.002823 |
| TwoPL baseline | `twopl_irt_online_baseline` | 0.800754 | 0.839229 | 0.896688 | 0.000000 |

Interpretation:
- Bias-calibrated grouped+TwoPL gets closer to TwoPL at `q=1000` but still does not surpass it on BA.
- Grouped models show strong AUROC/accuracy behavior, but BA at threshold 0.5 remains harder to optimize at high budget.

## RAM/runtime notes
- Runs were kept single-threaded for stability (`OMP_NUM_THREADS=1`, etc.).
- RAM was monitored during heavy sweeps.
- Typical usage was moderate; short peaks reached ~11-12 GiB / 15 GiB during the largest sweeps.
- No swap is configured; large sweeps were split into smaller chunks when needed.

## Artifacts
- Combined focused summary:
  - `reports/model_improvement_fasttext/grouped_residual_irt_focus_round2/focus_round2_summary_all.csv`
  - `reports/model_improvement_fasttext/grouped_residual_irt_focus_round2/focus_round2_grouped_only.csv`
  - `reports/model_improvement_fasttext/grouped_residual_irt_focus_round2/focus_round2_grouped_twopl_ensembles.csv`
- Final q=1000 bias analysis:
  - `reports/model_improvement_fasttext/grouped_residual_irt_q1000_bias_final/q1000_bias_candidates_all.csv`
  - `reports/model_improvement_fasttext/grouped_residual_irt_q1000_bias_final/q1000_bias_top60.csv`
  - `reports/model_improvement_fasttext/grouped_residual_irt_q1000_bias_final/q1000_bias_best_summary.csv`
