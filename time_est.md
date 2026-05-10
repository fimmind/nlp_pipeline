# Time Estimate

Evaluation setup:
- Models: top neural and top non-neural practical models from prior benchmarks.
- Budgets: 100, 1000, 2000 observed input words.
- Users: first 8 users from `responses_static`.
- Query sequence: `user_discriminative` fixed sequence.
- Features: `rich` feature set.
- Timing: mean `runtime_seconds` reported by the benchmark loop at each budget (incremental state updates).

## Results
| model | q | mean_time_seconds | mean_accuracy | mean_balanced_accuracy |
|---|---:|---:|---:|---:|
| neural_n2c_svd_ftkernel_hybrid_w20_55_25 | 100 | 0.9020 | 0.7590 | 0.7834 |
| neural_n2c_svd_ftkernel_hybrid_w20_55_25 | 1000 | 2.0762 | 0.7896 | 0.7902 |
| neural_n2c_svd_ftkernel_hybrid_w20_55_25 | 2000 | 3.5129 | 0.8049 | 0.8105 |
| refined_q1000_raw_w290_500_130_080_bias_p015 | 100 | 0.8837 | 0.7920 | 0.7802 |
| refined_q1000_raw_w290_500_130_080_bias_p015 | 1000 | 1.7748 | 0.8066 | 0.7535 |
| refined_q1000_raw_w290_500_130_080_bias_p015 | 2000 | 2.6797 | 0.8106 | 0.7641 |

## Per-Model View
- `neural_n2c_svd_ftkernel_hybrid_w20_55_25`
  - q=100: time=0.9020s, acc=0.7590, bal_acc=0.7834
  - q=1000: time=2.0762s, acc=0.7896, bal_acc=0.7902
  - q=2000: time=3.5129s, acc=0.8049, bal_acc=0.8105
- `refined_q1000_raw_w290_500_130_080_bias_p015`
  - q=100: time=0.8837s, acc=0.7920, bal_acc=0.7802
  - q=1000: time=1.7748s, acc=0.8066, bal_acc=0.7535
  - q=2000: time=2.6797s, acc=0.8106, bal_acc=0.7641

Source files:
- `reports/time_est/time_est_raw.csv`
- `reports/time_est/time_est_summary.csv`

## Rasch (Analogous Benchmark)
| model | q | mean_time_seconds | mean_accuracy | mean_balanced_accuracy |
|---|---:|---:|---:|---:|
| rasch_highbudget_var25 | 100 | 0.8428 | 0.7854 | 0.7761 |
| rasch_highbudget_var25 | 1000 | 1.6754 | 0.8077 | 0.7609 |
| rasch_highbudget_var25 | 2000 | 2.5140 | 0.8147 | 0.7757 |

Rasch source files:
- `reports/time_est/time_est_rasch_raw.csv`
- `reports/time_est/time_est_rasch_summary.csv`
