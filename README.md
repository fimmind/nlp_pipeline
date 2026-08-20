# Vocabulary Knowledge Prediction

Benchmark code and experiment outputs for online vocabulary knowledge estimation.

## Before Deleting a Local Clone

Ignored files are not restored by `git clone`. Before deleting the repository:

1. Confirm that all source changes and reports are committed and pushed:

```bash
git status -sb
git push origin master
```

2. Back up local state that cannot be regenerated:

- `data/user_profiles/`: quiz answers and observed known/unknown labels for local users
- `.env` and `.env.*`: local configuration or credentials, if present

For example:

```bash
tar -czf vocabulary-user-profiles.tar.gz data/user_profiles
```

Copy any `.env` files separately to an appropriate secure location; do not add credentials to Git.

Some profile files may already be tracked from older commits. Check with:

```bash
git ls-files data/user_profiles
```

Everything else currently ignored by this repository is either an external download, generated data, a cache, or a local development environment and is covered below.

## Reproducing Ignored Files

This repository intentionally excludes dependency caches, large external downloads, and generated normalized data products. The committed files keep the source code, tests, docs, compact raw inputs, and produced model/report outputs.

Run commands from the repository root.

### Python Environment

Desired location:

- `.venv/`

Source:

- Python packages declared in `pyproject.toml`
- Package indexes used by `pip`, normally PyPI

Recreate:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

### Duolingo HLR Learning Traces

Desired location:

- `data/raw/duolingo_hlr/learning_traces.csv.gz`

Exact source:

- Dataset: `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME`
- File: `settles.acl16.learning_traces.13m.csv.gz`
- File persistent ID: `doi:10.7910/DVN/N8XJME/UEPJVH`
- Direct API download: `https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/N8XJME/UEPJVH`
- Expected size: `379004009` bytes
- MD5: `0a1cae5eb7ad4b0bd9c0de91d74fcced`

Retrieve:

```bash
mkdir -p data/raw/duolingo_hlr
curl -L --fail \
  -o data/raw/duolingo_hlr/learning_traces.csv.gz \
  'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/N8XJME/UEPJVH'
```

`scripts/prepare_data.py` consumes this file automatically from the desired location. It can also copy it from another local path:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --duolingo-raw /path/to/settles.acl16.learning_traces.13m.csv.gz \
  --embedding-backend hash
```

### fastText Wiki/News Vectors

Desired location:

- `data/raw/wiki-news-300d-1M.vec.zip`

Exact source:

- `https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip`

Retrieve:

```bash
mkdir -p data/raw
curl -L -C - --fail --progress-bar \
  -o data/raw/wiki-news-300d-1M.vec.zip \
  https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
```

Use it to regenerate processed embeddings:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend fasttext \
  --fasttext-model-path data/raw/wiki-news-300d-1M.vec.zip \
  --fasttext-lang en \
  --embedding-dim 300 \
  --synthetic-if-missing \
  --skip-downloads
```

### fastText Common Crawl Binary Model

Desired locations:

- `cc.en.300.bin.gz`
- `cc.en.300.bin`

Exact source:

- `https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz`

Retrieve manually:

```bash
curl -L -C - --fail --progress-bar \
  -o cc.en.300.bin.gz \
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -dk cc.en.300.bin.gz
```

Retrieve through the existing data-preparation script:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend fasttext \
  --fasttext-lang en \
  --download-fasttext \
  --embedding-dim 300 \
  --synthetic-if-missing
```

That command calls `fasttext.util.download_model("en", if_exists="ignore")`, which downloads `cc.en.300.bin.gz` from the fastText Common Crawl URL and extracts `cc.en.300.bin`.

### Native fastText 12-Dimensional Benchmark Data (`data_ft12`)

`data_ft12/` is an isolated derivative used to benchmark soft groups formed directly from a 12-dimensional fastText model. It is not a separate source dataset. The preparation script loads `cc.en.300.bin` and calls `fasttext.util.reduce_model(model, 12)` before extracting word vectors.

Prerequisites:

- restore `cc.en.300.bin` as described above
- restore the ignored Duolingo trace file if exact parity with the existing 31,276-word benchmark inventory is required
- create the Python environment

Recreate the directory:

```bash
mkdir -p data_ft12/raw/frequency_sources
cp "data/raw/Responses L2 English speakers to 62 thousand words.xlsx" data_ft12/raw/
cp data/raw/frequency_sources/subtlex_word_frequencies_index.json \
  data_ft12/raw/frequency_sources/

.venv/bin/python -u scripts/prepare_data.py \
  --data-dir data_ft12 \
  --ehara-raw data/raw/ehara_esl_vocab/responses_raw.csv \
  --duolingo-raw data/raw/duolingo_hlr/learning_traces.csv.gz \
  --embedding-backend fasttext \
  --fasttext-model-path cc.en.300.bin \
  --fasttext-lang en \
  --embedding-dim 12 \
  --skip-downloads \
  --seed 42
```

Expected embedding metadata includes:

- `embedding_backend: fasttext`
- `original_dimension: 300`
- `dimension: 12`

The L2 grouped-strategy benchmark can then be reproduced with:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
.venv/bin/python scripts/run_l2_grouped_strategy_experiments.py \
  --data-dir data_ft12 \
  --out-dir reports/model_improvement_fasttext/l2_grouped_residual_native_fasttext12_smoke \
  --feature-set fasttext_only \
  --budgets 100,200,1000 \
  --groups 12 \
  --strategies native_fasttext_simplex,kmeans_fasttext,reduced_fasttext_simplex \
  --reduced-dims 12 \
  --temperatures 0.10 \
  --residual-priors 1.00 \
  --embedding-dim 12 \
  --seed 42
```

### Generated Processed Data

Desired locations:

- `data/processed/responses_static.csv`
- `data/processed/responses_temporal.csv`
- `data/processed/words.csv`
- `data/processed/frequency.csv`
- `data/processed/embeddings.npy`
- `data/processed/embeddings_metadata.json`
- `data/splits/static_leave_one_user_out.json`
- `data/splits/static_validation_users.json`
- `data/splits/cold_word_split.json`
- `data/DATASET_CARD.json`

Sources:

- committed raw static input: `data/raw/ehara_esl_vocab/responses_raw.csv`
- committed raw frequency input: `data/raw/frequency_sources/subtlex_word_frequencies_index.json`
- committed L2 workbook: `data/raw/Responses L2 English speakers to 62 thousand words.xlsx`
- optional restored Duolingo HLR file: `data/raw/duolingo_hlr/learning_traces.csv.gz`
- optional restored fastText source: `data/raw/wiki-news-300d-1M.vec.zip` or `cc.en.300.bin`

Recreate current fastText-style processed data after restoring `data/raw/wiki-news-300d-1M.vec.zip`:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend fasttext \
  --fasttext-model-path data/raw/wiki-news-300d-1M.vec.zip \
  --fasttext-lang en \
  --embedding-dim 300 \
  --synthetic-if-missing \
  --skip-downloads
```

Recreate a small deterministic smoke-test dataset without external downloads:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend hash \
  --skip-downloads \
  --synthetic-if-missing
```

### Workbook-Only Site Data (`data/processed/site_data`)

If you need site-focused processed data where words/difficulty come from the L2 workbook and soft groups come from response patterns across users, use:

```bash
.venv/bin/python scripts/generate_site_data_from_l2_workbook.py \
  --workbook "data/raw/Responses L2 English speakers to 62 thousand words.xlsx" \
  --out-dir data/processed/site_data \
  --ehara-raw data/raw/ehara_esl_vocab/responses_raw.csv \
  --evkd1-raw data/raw/evkd1/responses_raw.csv \
  --seed 42
```

This generator depends on:

- `data/raw/Responses L2 English speakers to 62 thousand words.xlsx` (sheet `Words`)
- columns `spelling` and `accuracy`
- `data/raw/ehara_esl_vocab/responses_raw.csv` (required for response-pattern grouping)
- `data/raw/evkd1/responses_raw.csv` (optional; used when present)

It writes:

- `data/processed/site_data/words.csv`
- `data/processed/site_data/difficulties.csv`
- `data/processed/site_data/grouped_residual_q_g12_seed42.csv`
- `data/processed/site_data/metadata.json`

Notes:

- `difficulties.csv` includes only word identity fields plus `accuracy` (no external frequency databases).
- Groupings are regenerated from response patterns across users using grouped residual IRT `Response12` grouping logic with `G=12`.
- Legacy `data/processed/grouped_residual_q_g12_seed42.csv` and `data/processed/grouped_residual_q_g16_seed42.csv` files are not consumed by current code. They do not need to be restored; use the supported `data/processed/site_data` G=12 export above.

### Vocabulary CLI Cache

Desired location:

- `data/cache/vocab_book_cli/`

This directory contains fitted-model, latent-state, probability, and preprocessed-book caches. It is disposable and should not be backed up. Running `scripts/vocab_book_cli.py` recreates the required entries from processed data, the selected book, and the user profile.

### User Profiles

Desired location:

- `data/user_profiles/`

Profiles contain user-supplied quiz answers and later known/unknown observations. They cannot be inferred from the model or regenerated from raw datasets. Back up any untracked profile files before deleting the local clone. After cloning, restore the files to the same directory; runtime caches will be rebuilt automatically.

### Runtime Caches and Build Artifacts

Ignored locations:

- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- `src/vocab_benchmark.egg-info/`
- `.mypy_cache/`
- `.ruff_cache/`
- `build/`
- `dist/`

Sources:

- Python, pytest, setuptools, and local package execution.

Recreate:

```bash
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```

These files are not required for reproducing results and should not be committed.

## Practical Book Vocabulary CLI

The repository includes a practical script that asks the user to mark 100 words as known/unknown, infers word-level knowledge with the current best model, and analyzes books in `data/example_texts/`.

Script:

- `scripts/vocab_book_cli.py`

### List Available Models

```bash
.venv/bin/python scripts/vocab_book_cli.py --list-models
```

The 100-word profile is model-agnostic and reusable across models. You can run the test once, then compare different models without retaking it.

### Choose Model (`--model`)

Use `--model <name>` to switch the estimator used for inference.

Practical guidance:

| Model | Type | Relative speed | Notes |
|---|---|---|---|
| `rasch` | non-neural | fastest | Good baseline for quick checks. |
| `twopl` | non-neural | very fast | Slightly richer than Rasch. |
| `rasch_twopl_vote_user` | non-neural ensemble | medium | Strong practical blend for mixed budgets. |
| `best_high_budget` | non-neural ensemble | medium | Best fixed blend for larger observed budgets. |
| `best_adaptive` | non-neural ensemble | medium | Current default/best practical model. |
| `best_grouped_irt_model` | grouped IRT | medium | Best grouped-residual IRT variant in current benchmarks. |

Example with a fast non-neural model:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/vocab_book_cli.py \
  --profile your_name \
  --model rasch \
  --book AiW.txt
```

### First Run (Interactive)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/vocab_book_cli.py \
  --profile your_name \
  --model best_adaptive \
  --book AiW.txt
```

What happens:

1. The script prints the 100-word test list.
2. You answer each item with `y/n` (or `known/unknown`, `1/0`).
3. The profile is saved to `data/user_profiles/your_name.json`.
4. The selected book is analyzed using the inferred vocabulary knowledge.

### Reuse Saved Profile On Another Book

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/vocab_book_cli.py \
  --profile your_name \
  --model svd \
  --book "The hitchhikers guide to the galaxy - Douglas Adams.txt"
```

If `--retake-test` is not provided, the saved profile is loaded and reused.

### Retake The 100-Word Test

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/vocab_book_cli.py \
  --profile your_name \
  --model best_adaptive \
  --retake-test \
  --book "The Great Gatsby.txt"
```

### Non-Interactive / Automation Run

Use `--answer-string` with exactly 100 characters from `y/n/1/0/k/u`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/vocab_book_cli.py \
  --profile smoke_user \
  --model rasch \
  --retake-test \
  --answer-string "$(printf 'y%.0s' {1..100})" \
  --book AiW.txt
```

### Add Known/Unknown Words Without Retaking Test

You can append or update labels in an existing profile:

```bash
scripts/vocab_book_cli.sh \
  --profile your_name \
  --model rasch \
  --book AiW.txt \
  --add-words "kinship=known,mixture=unknown"
```

Interactive add/update mode:

```bash
scripts/vocab_book_cli.sh \
  --profile your_name \
  --model rasch \
  --book AiW.txt \
  --add-words-interactive
```

When profile labels are changed, stale probability caches are invalidated automatically and rebuilt consistently.

### Query Specific Words

Use `--query-words` with comma-separated words:

```bash
scripts/vocab_book_cli.sh \
  --profile your_name \
  --model rasch \
  --book AiW.txt \
  --query-words "rabbit,alice,kinship"
```

For each queried word the script prints:

1. model prediction (`p_known` and known/unknown decision),
2. whether the word was observed in the profile before, including the last observed state,
3. when `--book` is provided: sentences containing the queried word, sorted by predicted unknown-word count (least to most).  
Alongside each sentence it prints unknown words in that sentence and their model predictions.

When `--query-words` is provided, the script suppresses the general book-analysis sections:
`Book Vocabulary Estimate`, random expected known/unknown word lists, and
`Sentences Expected To Have Exactly One Unknown Word`.

### Book Estimate Output

The script prints:

1. `Book Vocabulary Estimate` with unknown-token count and percentage.
2. 25 random in-book words expected known.
3. 25 random in-book words expected unknown.
4. Up to 10 sentences where exactly one in-vocabulary word is expected unknown.

Out-of-model-vocabulary tokens are discarded from unknown-word computations.
