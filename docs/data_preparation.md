# Data Preparation

## Smoke test with synthetic fallback

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend hash \
  --skip-downloads \
  --synthetic-if-missing
```

This always creates:

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

## Manual dataset placement

- Ehara static data:
  - `data/raw/ehara_esl_vocab/responses_raw.csv`
  - or pass `--ehara-raw /path/to/responses_raw.csv`
- EVKD1 static data:
  - `data/raw/evkd1/responses_raw.csv`
  - or pass `--evkd1-raw /path/to/responses_raw.csv`
- Duolingo HLR temporal data:
  - `data/raw/duolingo_hlr/learning_traces.csv.gz`
  - or pass `--duolingo-raw /path/to/learning_traces.csv.gz`

## fastText embeddings

Use hash embeddings for smoke tests. For stronger semantics:

```bash
python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend fasttext \
  --fasttext-lang en \
  --download-fasttext \
  --embedding-dim 300 \
  --synthetic-if-missing
```

If fastText loading/download fails, the script now fails instead of silently writing hash embeddings. Pass `--allow-hash-fallback` only for smoke tests where hash embeddings are acceptable.

For a lighter fastText artifact than `cc.en.300.bin`, download the official wiki/news vector zip and pass it explicitly:

```bash
curl -L -C - --fail --progress-bar \
  -o data/raw/wiki-news-300d-1M.vec.zip \
  https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

python -u scripts/prepare_data.py \
  --data-dir data \
  --embedding-backend fasttext \
  --fasttext-model-path data/raw/wiki-news-300d-1M.vec.zip \
  --fasttext-lang en \
  --embedding-dim 300 \
  --synthetic-if-missing \
  --skip-downloads
```

## Processed file semantics

- `responses_static.csv`: main benchmark labels (`user_id`, `word_id`, `word`, `label`)
- `responses_temporal.csv`: secondary temporal recall traces
- `words.csv`: one row per word aligned with embedding rows
- `frequency.csv`: `frequency` and `log_frequency` features
- `embeddings.npy`: `embeddings[i]` aligns with `words.csv.iloc[i]`
- split files: LOOU users, validation/test users, and cold-word holdout
