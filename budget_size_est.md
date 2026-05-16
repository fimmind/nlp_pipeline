# Budget Size Accuracy Estimate

Evaluation setup:
- Models: top 2 neural + top 2 non-neural practical candidates.
- Budgets: 10, 20, 30, 40, 50, 100 initial observed words.
- Users: first 8 users from `responses_static` (LOOU-style per-user evaluation).
- Query sequence: `user_discriminative` fixed sequence.
- Features: `rich` word feature set.
- Metric shown below: **raw accuracy** (mean over evaluated users/splits).
- State progression: **incremental**. Latent user state is updated only with newly added observations when moving from one budget to the next (no full rebuild per budget).

## Neural Models
| model | q=10 | q=20 | q=30 | q=40 | q=50 | q=100 |
|---|---:|---:|---:|---:|---:|---:|
| neural_n2c_svd_ftkernel_hybrid_w20_55_25 | 0.7621 | 0.7648 | 0.7612 | 0.7563 | 0.7618 | 0.7590 |
| neural_memory_mirt_n2c_rate35 | 0.7620 | 0.7638 | 0.7596 | 0.7567 | 0.7616 | 0.7609 |

## Non-Neural Models
| model | q=10 | q=20 | q=30 | q=40 | q=50 | q=100 |
|---|---:|---:|---:|---:|---:|---:|
| budget_adaptive_refined_raw_switch500 | 0.7944 | 0.7919 | 0.7836 | 0.7803 | 0.7866 | 0.7882 |
| refined_q1000_raw_w290_500_130_080_bias_p015 | 0.7974 | 0.7918 | 0.7849 | 0.7803 | 0.7901 | 0.7924 |

## Best Model Per Budget (Raw Accuracy)
| q | best_model | raw_accuracy | balanced_accuracy |
|---:|---|---:|---:|
| 10 | refined_q1000_raw_w290_500_130_080_bias_p015 | 0.7974 | 0.7685 |
| 20 | budget_adaptive_refined_raw_switch500 | 0.7919 | 0.7698 |
| 30 | refined_q1000_raw_w290_500_130_080_bias_p015 | 0.7849 | 0.7728 |
| 40 | budget_adaptive_refined_raw_switch500 | 0.7803 | 0.7754 |
| 50 | refined_q1000_raw_w290_500_130_080_bias_p015 | 0.7901 | 0.7736 |
| 100 | refined_q1000_raw_w290_500_130_080_bias_p015 | 0.7924 | 0.7806 |

Source files:
- `reports/budget_size_est/top_neural_nonneural_raw_accuracy_q10_100.csv`
- `reports/budget_size_est/top_neural_nonneural_summary_q10_100.csv`

## Rasch Model (Same Protocol)
| model | q=10 | q=20 | q=30 | q=40 | q=50 | q=100 |
|---|---:|---:|---:|---:|---:|---:|
| rasch_highbudget_var25 | 0.7999 | 0.7912 | 0.7908 | 0.7803 | 0.7912 | 0.7854 |

Rasch vs best previously tested model at each budget (raw accuracy gap):
| q | Rasch | Best previous | Gap (Rasch - Best) |
|---:|---:|---:|---:|
| 10 | 0.7999 | 0.7974 | +0.0025 |
| 20 | 0.7912 | 0.7919 | -0.0007 |
| 30 | 0.7908 | 0.7849 | +0.0059 |
| 40 | 0.7803 | 0.7803 | +0.0000 |
| 50 | 0.7912 | 0.7901 | +0.0011 |
| 100 | 0.7854 | 0.7924 | -0.0071 |

Rasch source files:
- `reports/budget_size_est/rasch_raw_accuracy_q10_100.csv`
- `reports/budget_size_est/rasch_summary_q10_100.csv`

## Current Best Model: Balanced Accuracy vs Query Budget
Evaluation setup:
- Model: `response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk` (current best grouped residual IRT in practical pipeline).
- Protocol: `within_user_completion` (infer the rest of the same learner's vocabulary after `q` answers).
- Budgets: `q = 10, 20, 30, 50, 100, 200`.
- Query sequence: fixed global `user_discriminative` sequence (deterministic seed `42`).
- Repeats: `3`.
- Features: `rich`.
- State progression: incremental updates only (state is not rebuilt from scratch between budgets).

| q | balanced_accuracy | accuracy | nll | brier | auroc |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.7588 | 0.7726 | 0.5008 | 0.1636 | 0.8673 |
| 20 | 0.7888 | 0.7736 | 0.5071 | 0.1655 | 0.8802 |
| 30 | 0.7942 | 0.7982 | 0.4767 | 0.1529 | 0.8878 |
| 50 | 0.8085 | 0.8112 | 0.4526 | 0.1435 | 0.8952 |
| 100 | 0.8284 | 0.8366 | 0.4103 | 0.1274 | 0.8992 |
| 200 | 0.8193 | 0.8296 | 0.3995 | 0.1259 | 0.8858 |

Notes:
- Balanced accuracy improves strongly from `q=10` to `q=100` (+0.0696 absolute), then slightly drops at `q=200` in this run.
- Peak in this sweep is at `q=100` with BA `0.8284`.

Source files:
- `reports/budget_size_est/current_best_budget_sweep_raw.csv`
- `reports/budget_size_est/current_best_budget_sweep_summary.csv`
