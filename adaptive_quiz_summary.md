# Adaptive Quiz Benchmark Summary

## Scope and Objective
This report summarizes recent adaptive quiz experiments for **online vocabulary inference** with the grouped residual IRT model family. The evaluation question is:

> How well can the model infer the rest of the same learner's vocabulary after observing `q` answers?

Primary metric: **Balanced Accuracy** on each learner's still-hidden labeled words after `q` observed answers.

Data sources used in this summary:
- `reports/adaptive_quiz/adaptive_quiz_effect_raw.csv`
- `reports/adaptive_quiz/adaptive_quiz_effect_summary.csv`

---

## Experimental Setup

### Model
The benchmark was run with grouped residual IRT estimator configuration:
- `tau_theta=2.0`
- `tau_delta=1.6`
- `gate_c=12.0`
- `n_groups=12`
- observed-label threshold optimization/shrink enabled in estimator config

### Population and labels
- 16 users (the available static labeled dataset slice used by this benchmark run)
- For each user, only their labeled words participate.

### Protocol
For each user and strategy:
1. Start with empty user state.
2. Ask one word at a time up to max budget.
3. After each observed response, update user state incrementally.
4. At target budgets (`q=50`, `q=100` in this run), evaluate Balanced Accuracy on remaining hidden labeled words.

### Compared query strategies
- `static_det`: fixed deterministic sequence
- `static_semirand`: static sequence with semi-randomization
- `adaptive_entropy`: choose words closest to model decision boundary (`p≈0.5`)
- `adaptive_uncertainty`: choose by estimator uncertainty directly
- `adaptive_stoch_top3`: stochastic pick among top-3 entropy candidates

---

## Headline Results

### Mean Balanced Accuracy (higher is better)

At `q=50`:
1. `adaptive_uncertainty`: **0.8570**
2. `adaptive_entropy`: 0.8562
3. `adaptive_stoch_top3`: 0.8504
4. `static_semirand`: 0.8082
5. `static_det`: 0.7994

At `q=100`:
1. `adaptive_uncertainty`: **0.8737**
2. `adaptive_entropy`: 0.8729
3. `adaptive_stoch_top3`: 0.8725
4. `static_det`: 0.8298
5. `static_semirand`: 0.8290

---

## Statistical Detail (Per-user variability)

Sample size for each strategy/budget: `n=16` users.

### q=50
- `adaptive_uncertainty`: mean `0.8570`, std `0.0703`, 95% CI `±0.0344`, min `0.7157`, max `0.9475`
- `adaptive_entropy`: mean `0.8562`, std `0.0738`, 95% CI `±0.0361`
- `adaptive_stoch_top3`: mean `0.8504`, std `0.0762`, 95% CI `±0.0374`
- `static_semirand`: mean `0.8082`, std `0.0946`, 95% CI `±0.0464`
- `static_det`: mean `0.7994`, std `0.1034`, 95% CI `±0.0507`

### q=100
- `adaptive_uncertainty`: mean `0.8737`, std `0.0754`, 95% CI `±0.0369`, min `0.7403`, max `0.9696`
- `adaptive_entropy`: mean `0.8729`, std `0.0799`, 95% CI `±0.0392`
- `adaptive_stoch_top3`: mean `0.8725`, std `0.0832`, 95% CI `±0.0408`
- `static_det`: mean `0.8298`, std `0.0966`, 95% CI `±0.0474`
- `static_semirand`: mean `0.8290`, std `0.0981`, 95% CI `±0.0481`

Key point: adaptive methods reduce variance and improve average accuracy relative to static methods.

---

## Pairwise Comparisons

Using per-user differences against the best mean strategy (`adaptive_uncertainty`):

### q=50
- vs `adaptive_entropy`: mean delta `+0.0008`, wins/ties/losses = `8/0/8`
- vs `adaptive_stoch_top3`: mean delta `+0.0067`, `10/0/6`
- vs `static_det`: mean delta `+0.0576`, `15/0/1`
- vs `static_semirand`: mean delta `+0.0488`, `15/0/1`

### q=100
- vs `adaptive_entropy`: mean delta `+0.0009`, `8/0/8`
- vs `adaptive_stoch_top3`: mean delta `+0.0012`, `7/0/9`
- vs `static_det`: mean delta `+0.0440`, `16/0/0`
- vs `static_semirand`: mean delta `+0.0448`, `16/0/0`

Interpretation:
- `adaptive_uncertainty`, `adaptive_entropy`, and `adaptive_stoch_top3` are very close at `q=100`.
- All adaptive strategies strongly dominate static baselines on this benchmark.

---

## Practical Conclusions

1. **Adaptive querying is clearly worthwhile** for grouped IRT online inference under this protocol.
2. **`adaptive_uncertainty` is the best default** right now:
   - top mean Balanced Accuracy at both evaluated budgets,
   - consistently strong pairwise outcomes vs static,
   - much simpler and typically cheaper than deeper lookahead strategies.
3. **Gains are largest at lower budgets** (q=50), where question efficiency matters most.

---

## Limits and Cautions

1. Dataset size is small (16 users), so ranking differences among the top 3 adaptive methods are narrow.
2. This report covers `q=50` and `q=100` from these artifacts; longer-budget behavior (`q=200+`) is not included here.
3. Runtime/RAM were not logged into these CSVs directly; separate profiling should accompany future heavy lookahead policy comparisons.

---

## Recommended Next Experiments

1. Add **runtime + peak RAM** logging per strategy in the same benchmark output table.
2. Run a compact comparison including:
   - `adaptive_uncertainty`
   - `adaptive_entropy`
   - a tuned lightweight one-step lookahead hybrid
   at `q={30,50,100,200}`.
3. Add **queries-to-target** analysis for thresholds (e.g., BA ≥ 0.85, 0.90) to quantify quiz length savings.
4. Report confidence intervals via bootstrap across users for stronger robustness checks.

---

## Recommendation for current CLI default
Use **`adaptive_uncertainty`** as the grouped-IRT default quiz strategy until a hybrid policy demonstrates materially higher BA at similar latency and RAM cost.
