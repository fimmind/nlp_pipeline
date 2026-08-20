# L2 Grouped Residual IRT (Non-HDBSCAN) Analysis

## Scope
This report compares two grouping strategies under the same evaluation setup:
- `kmeans_fasttext`
- `wordnet_supersense`

HDBSCAN was excluded from this analysis due stability/memory issues in the current environment.

## Data/Model Constraints
- Difficulty prior: `accuracy` from L2 dataset (`data/raw/Responses L2 English speakers to 62 thousand words.xlsx` via processed `frequency.csv` alignment).
- Grouping inputs:
  - `kmeans_fasttext`: fastText embedding space.
  - `wordnet_supersense`: WordNet supersense-derived vectors.
- Evaluation labels: `responses_static` (LOOU protocol), `max_users=8`.
- Budgets: `q in {100, 200, 1000}`.

## Generated Artifacts
- Combined summary: `reports/model_improvement_fasttext/l2_grouped_residual_non_hdbscan/combined_summary.csv`
- Best per strategy/q: `reports/model_improvement_fasttext/l2_grouped_residual_non_hdbscan/best_by_family_and_q.csv`
- Top models per budget:
  - `top_q100.csv`
  - `top_q200.csv`
  - `top_q1000.csv`

## Best Overall Models
- `q=100`: `l2_grouped_kmeans_fasttext_g12_t025_rp100` — BA `0.6598`
- `q=200`: `l2_grouped_kmeans_fasttext_g12_t025_rp100` — BA `0.6399`
- `q=1000`: `l2_grouped_kmeans_fasttext_g12_t025_rp100` — BA `0.6293`

## Best WordNet Models
- `q=100`: `l2_grouped_wordnet_supersense_all_synsets_g12_t010_rp100` — BA `0.6586`
- `q=200`: `l2_grouped_wordnet_supersense_all_synsets_g16_t010_rp100` — BA `0.6377`
- `q=1000`: `l2_grouped_wordnet_supersense_all_synsets_g12_t010_rp100` — BA `0.6268`

## Comparative Findings
- k-means fastText slightly outperformed WordNet supersense at all tested budgets.
- Best-performing cluster count was consistently around `g=12` (k-means).
- WordNet mode `all_synsets` slightly outperformed `first_synset`.
- Within tested ranges, temperature `t=0.25` performed slightly better than `t=0.10` for k-means.

## Caveat
Absolute BA values are below prior best hybrid/ensemble models in this repository. These runs isolate the requested strategy family under strict L2-accuracy grouped-residual assumptions.
