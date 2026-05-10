# Basic Rasch From Accuracy (No Embeddings)

## Model Definition

Estimator: `BasicRaschFromAccuracyEstimator`  
Implementation: `src/vocab_benchmark/estimators/irt.py`

This estimator is a fixed-difficulty Rasch model where item difficulty is taken directly from the `accuracy` field (percentage of learners that know the word), sourced from:

- `data/raw/Responses L2 English speakers to 62 thousand words.xlsx`
- tab: `Words`
- column: `accuracy`

In runtime, the estimator reads `accuracy` from `data/processed/frequency.csv` (the processed projection of that source) and sets:

- `p_i = clip(accuracy_i, 1e-4, 1-1e-4)` (auto-converts `0..100` to `0..1` when needed)
- `b_i = -logit(p_i)`

No response-based item pretraining is used.

## Embedding Independence

This model does **not** use fastText (or any other embedding) values for item difficulty or prediction features.

- It only needs word indices and observed user labels.
- `word_features` are used only to get `n_words` shape for interface compatibility.
- Difficulty is derived exclusively from the `accuracy` column.

So it supports the full word inventory present in the processed frequency table, independently of embedding semantics.

## 16-User Evaluation (LOOU)

Protocol:

- Dataset: static 16-user set
- Split: leave-one-user-out over all 16 users
- Query sequence: `user_discriminative`
- Budgets: `q in {100, 1000, 2000}`
- Single-threaded eval

Results:

| estimator | q | accuracy | balanced_accuracy | average_precision_known | average_precision_unknown | runtime_seconds |
|---|---:|---:|---:|---:|---:|---:|
| basic_rasch_from_accuracy | 100 | 0.801457 | 0.736445 | 0.845434 | 0.773044 | 0.872759 |
| basic_rasch_from_accuracy | 1000 | 0.814046 | 0.724037 | 0.857833 | 0.789470 | 1.739908 |
| basic_rasch_from_accuracy | 2000 | 0.816395 | 0.711501 | 0.870307 | 0.803532 | 2.622707 |

Artifacts:

- `reports/time_est/basic_rasch_accuracy16_raw.csv`
- `reports/time_est/basic_rasch_accuracy16_summary.csv`
