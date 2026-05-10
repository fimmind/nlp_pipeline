# Vocabulary Knowledge Prediction

Benchmark code and experiment outputs for online vocabulary knowledge estimation.

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
| `vote` | non-neural | very fast | User-similarity vote model. |
| `svd` | non-neural | fast | Strong collaborative latent model. |
| `rasch_vote` | non-neural | fast | Lightweight hybrid. |
| `user_logreg` | non-neural | medium | Fits per-user logistic model from observed answers. |
| `fasttext_kernel` | non-neural | medium | Semantic kernel logistic model over embeddings. |
| `best_high_budget` | non-neural ensemble | medium | Best fixed blend for larger observed budgets. |
| `best_adaptive` | non-neural ensemble | medium | Current default/best practical model. |

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

### Book Estimate Output

The script prints:

1. `Book Vocabulary Estimate` with unknown-token count and percentage.
2. 25 random in-book words expected known.
3. 25 random in-book words expected unknown.
4. Up to 10 sentences where exactly one in-vocabulary word is expected unknown.

Out-of-model-vocabulary tokens are discarded from unknown-word computations.
