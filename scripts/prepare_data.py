#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


STATIC_COLUMNS: list[str] = [
    "user_id",
    "word_id",
    "word",
    "label",
    "raw_score",
    "timestamp",
    "source",
    "language",
]
TEMPORAL_COLUMNS: list[str] = [
    "user_id",
    "word_id",
    "word",
    "lemma",
    "pos",
    "morphology",
    "label",
    "p_recall",
    "timestamp",
    "delta",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
    "learning_language",
    "ui_language",
    "source",
]
WORDS_COLUMNS: list[str] = [
    "word_id",
    "word",
    "lemma",
    "pos",
    "morphology",
    "language",
    "source",
    "length",
]
FREQUENCY_COLUMNS: list[str] = [
    "word_id",
    "word",
    "language",
    "frequency",
    "log_frequency",
    "wordfreq_zipf",
    "subtlex_us_count",
    "log_frequency_subtlex_us",
    "subtlex_us_rank",
    "subtlex_us_rank_percentile",
    "frequency_rank",
    "frequency_rank_percentile",
    "frequency_band",
    "accuracy",
    "acc_L2",
    "rank_L2",
    "nobs_L2",
    "acc_L1",
    "rank_L1",
    "diff_L1_L2",
]


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    splits_dir: Path
    raw_ehara_dir: Path
    raw_evkd1_dir: Path
    raw_duolingo_dir: Path


def ensure_paths(data_dir: Path) -> Paths:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    paths = Paths(
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        raw_ehara_dir=raw_dir / "ehara_esl_vocab",
        raw_evkd1_dir=raw_dir / "evkd1",
        raw_duolingo_dir=raw_dir / "duolingo_hlr",
    )
    for path in [
        paths.data_dir,
        paths.raw_dir,
        paths.processed_dir,
        paths.splits_dir,
        paths.raw_ehara_dir,
        paths.raw_evkd1_dir,
        paths.raw_duolingo_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_colname(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {normalize_colname(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized:
            return normalized[key]
    return None


def stable_word_id(word: str) -> str:
    digest = hashlib.sha1(str(word).strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"w_{digest}"


def coerce_binary_label(series: pd.Series, raw_score: pd.Series | None, binarization: str) -> pd.Series:
    if raw_score is not None:
        numeric_score = pd.to_numeric(raw_score, errors="coerce")
        if numeric_score.notna().any():
            max_score = numeric_score.max()
            if binarization == "relaxed":
                return (numeric_score >= max_score - 1).astype("Int64")
            return (numeric_score == max_score).astype("Int64")
    values = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "y", "known", "know", "correct"}
    false_values = {"0", "false", "no", "n", "unknown", "dont_know", "don't know", "incorrect"}
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out[values.isin(true_values)] = 1
    out[values.isin(false_values)] = 0
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        unique = set(numeric.dropna().unique().tolist())
        if unique <= {0, 1}:
            out[numeric == 1] = 1
            out[numeric == 0] = 0
        else:
            max_score = numeric.max()
            if binarization == "relaxed":
                out[numeric >= max_score - 1] = 1
                out[numeric < max_score - 1] = 0
            else:
                out[numeric == max_score] = 1
                out[numeric < max_score] = 0
    return out


def normalize_static_dataset(raw_path: Path, source: str, binarization: str) -> pd.DataFrame:
    log(f"[prepare] static source={source} path={raw_path}")
    df = pd.read_csv(raw_path)
    user_col = find_col(df, ["user_id", "userid", "user", "learner_id", "learner", "student_id"])
    word_col = find_col(df, ["word", "surface", "surface_form", "vocabulary", "item", "token"])
    word_id_col = find_col(df, ["word_id", "item_id", "vocab_id", "vocabulary_id", "lexeme_id"])
    label_col = find_col(df, ["label", "known", "is_known", "answer", "correct", "binary_label"])
    score_col = find_col(df, ["raw_score", "score", "rating", "knowledge", "knowledge_score"])
    timestamp_col = find_col(df, ["timestamp", "time", "created_at", "answered_at"])
    if user_col is None:
        raise ValueError(f"{raw_path}: cannot infer user_id column")
    if word_col is None and word_id_col is None:
        raise ValueError(f"{raw_path}: cannot infer word/word_id column")
    if label_col is None and score_col is None:
        raise ValueError(f"{raw_path}: cannot infer label/raw_score column")

    out = pd.DataFrame()
    out["user_id"] = df[user_col].astype(str)
    if word_id_col is not None:
        out["word_id"] = df[word_id_col].astype(str)
    else:
        out["word_id"] = df[word_col].astype(str).map(stable_word_id)
    out["word"] = df[word_col].astype(str) if word_col is not None else out["word_id"]
    raw_score = df[score_col] if score_col is not None else None
    if label_col is not None:
        out["label"] = coerce_binary_label(df[label_col], raw_score, binarization)
    else:
        assert raw_score is not None
        out["label"] = coerce_binary_label(raw_score, raw_score, binarization)
    out["raw_score"] = raw_score if raw_score is not None else pd.NA
    out["timestamp"] = df[timestamp_col] if timestamp_col is not None else pd.NA
    out["source"] = source
    out["language"] = "en"
    out = out.dropna(subset=["user_id", "word_id", "label"])
    out["label"] = out["label"].astype(int)
    out = out[STATIC_COLUMNS].drop_duplicates(subset=["source", "user_id", "word_id"], keep="first")
    return out


def parse_duolingo_lexeme(lexeme_string: str) -> tuple[str, str | None, str | None, str | None]:
    text = str(lexeme_string)
    match = re.match(r"([^/]+)/([^<]+)<([^>]+)>(.*)", text)
    if match is None:
        return text, None, None, None
    surface, lemma, pos, rest = match.groups()
    morph = " ".join(re.findall(r"<([^>]+)>", rest)) or None
    return surface, lemma, pos, morph


def normalize_duolingo_dataset(raw_path: Path, max_rows: int | None) -> pd.DataFrame:
    log(f"[prepare] temporal source=duolingo path={raw_path}")
    compression = "gzip" if raw_path.suffix == ".gz" else None
    df = pd.read_csv(raw_path, compression=compression, nrows=max_rows)
    required = ["user_id", "lexeme_id", "lexeme_string"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"{raw_path}: missing required column {column}")
    out = pd.DataFrame()
    out["user_id"] = df["user_id"].astype(str)
    out["word_id"] = df["lexeme_id"].astype(str)
    parsed = df["lexeme_string"].astype(str).map(parse_duolingo_lexeme)
    out["word"] = parsed.map(lambda x: x[0])
    out["lemma"] = parsed.map(lambda x: x[1])
    out["pos"] = parsed.map(lambda x: x[2])
    out["morphology"] = parsed.map(lambda x: x[3])
    if "p_recall" in df.columns:
        out["p_recall"] = pd.to_numeric(df["p_recall"], errors="coerce")
        out["label"] = (out["p_recall"] >= 0.5).astype("Int64")
    elif "session_seen" in df.columns and "session_correct" in df.columns:
        seen = pd.to_numeric(df["session_seen"], errors="coerce")
        correct = pd.to_numeric(df["session_correct"], errors="coerce")
        out["p_recall"] = correct / seen.replace(0, np.nan)
        out["label"] = (out["p_recall"] >= 0.5).astype("Int64")
    else:
        out["p_recall"] = pd.NA
        out["label"] = pd.NA
    for column in ["timestamp", "delta", "history_seen", "history_correct", "session_seen", "session_correct"]:
        out[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    out["learning_language"] = df["learning_language"].astype(str) if "learning_language" in df.columns else "unknown"
    out["ui_language"] = df["ui_language"].astype(str) if "ui_language" in df.columns else "unknown"
    out["source"] = "duolingo_hlr"
    out = out[TEMPORAL_COLUMNS]
    return out


def generate_synthetic_static_dataset(n_users: int, n_words: int, embedding_dim: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_domains = min(8, max(3, embedding_dim // 16))
    word_ids = [f"syn_w_{i}" for i in range(n_words)]
    words = [f"word_{i}" for i in range(n_words)]
    domain_id = rng.integers(0, n_domains, size=n_words)
    domain_centers = rng.normal(0.0, 1.0, size=(n_domains, embedding_dim)).astype(np.float32)
    difficulties = rng.normal(0.0, 1.0, size=n_words) + 0.5 * rng.normal(0.0, 1.0, size=n_words)
    embeddings = np.zeros((n_words, embedding_dim), dtype=np.float32)
    for i in range(n_words):
        vec = domain_centers[domain_id[i]] + 0.30 * rng.normal(0.0, 1.0, size=embedding_dim) - 0.03 * difficulties[i]
        norm = np.linalg.norm(vec)
        embeddings[i] = (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)
    user_ids = [f"syn_u_{i}" for i in range(n_users)]
    theta = rng.normal(0.0, 1.0, size=n_users)
    pref = rng.normal(0.0, 0.8, size=(n_users, n_domains))
    loading = rng.normal(0.0, 0.8, size=(n_words, n_domains))
    rows: list[dict[str, object]] = []
    for u_idx, user_id in enumerate(user_ids):
        logits = theta[u_idx] - difficulties + loading @ pref[u_idx] + rng.normal(0.0, 0.15, size=n_words)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        labels = rng.binomial(1, probs)
        for i in range(n_words):
            rows.append(
                {
                    "user_id": user_id,
                    "word_id": word_ids[i],
                    "word": words[i],
                    "label": int(labels[i]),
                    "raw_score": np.nan,
                    "timestamp": np.nan,
                    "source": "synthetic",
                    "language": "en",
                }
            )
    responses = pd.DataFrame(rows, columns=STATIC_COLUMNS)
    words_df = pd.DataFrame(
        {
            "word_id": word_ids,
            "word": words,
            "lemma": words,
            "pos": ["X"] * n_words,
            "morphology": [""] * n_words,
            "language": ["en"] * n_words,
            "source": ["synthetic"] * n_words,
            "length": [len(w) for w in words],
        }
    )
    latent_frequency = np.exp(-difficulties + rng.normal(0.0, 0.4, size=n_words))
    frequency = pd.DataFrame(
        {
            "word_id": word_ids,
            "word": words,
            "language": ["en"] * n_words,
            "frequency": latent_frequency.astype(float),
            "log_frequency": np.log1p(latent_frequency).astype(float),
        }
    )
    metadata = {
        "embedding_backend": "synthetic",
        "dimension": embedding_dim,
        "normalized_l2": True,
        "aligned_to": "processed/words.csv row order",
        "notes": "Synthetic benchmark data from latent ability/difficulty/domain model.",
    }
    return responses, words_df, frequency, embeddings, metadata


def _load_subtlex_us_counts(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "frequency_sources" / "subtlex_word_frequencies_index.json"
    if not path.exists():
        log(f"[warn] SUBTLEXus frequency file missing: {path}")
        return pd.DataFrame(columns=["word_key", "subtlex_us_count", "subtlex_us_rank", "subtlex_us_rank_percentile", "log_frequency_subtlex_us"])
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        word = str(row.get("word", "")).strip().lower()
        if word == "":
            continue
        count = pd.to_numeric(row.get("count", row.get("value", np.nan)), errors="coerce")
        parsed_rows.append({"word_key": word, "subtlex_us_count": count, "subtlex_us_rank": float(index + 1)})
    subtlex = pd.DataFrame(parsed_rows)
    if subtlex.empty:
        return pd.DataFrame(columns=["word_key", "subtlex_us_count", "subtlex_us_rank", "subtlex_us_rank_percentile", "log_frequency_subtlex_us"])
    subtlex = subtlex.dropna(subset=["subtlex_us_count"])
    subtlex = subtlex.drop_duplicates(subset=["word_key"], keep="first")
    subtlex["subtlex_us_rank_percentile"] = (subtlex["subtlex_us_rank"] - 1.0) / max(1, len(subtlex) - 1)
    per_billion = pd.to_numeric(subtlex["subtlex_us_count"], errors="coerce") / 51_000_000.0 * 1_000_000_000.0
    subtlex["log_frequency_subtlex_us"] = np.log10(per_billion.clip(lower=1e-12))
    log(f"[prepare] SUBTLEXus frequencies loaded rows={len(subtlex)}")
    return subtlex


def _rank_percentile_from_score(values: pd.Series) -> tuple[pd.Series, pd.Series, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    rank = numeric.rank(method="average", ascending=False)
    percentile = ((rank - 1.0) / max(1, len(numeric) - 1)).astype(float)
    bands = np.full(len(numeric), 999999, dtype=np.int32)
    rank_values = rank.to_numpy()
    for cutoff in [1000, 2000, 3000, 5000, 10000, 20000]:
        selected = rank_values <= cutoff
        bands[selected] = np.minimum(bands[selected], cutoff)
    return rank.astype(float), percentile, bands


def add_frequency_features(words: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    freq = words[["word_id", "word", "language"]].copy()
    freq["frequency"] = np.nan
    freq["log_frequency"] = np.nan
    freq["wordfreq_zipf"] = np.nan
    try:
        from wordfreq import zipf_frequency
        best_values: list[float] = []
        for row in freq.itertuples(index=False):
            lang = str(row.language).strip().lower()
            lang = "en" if lang in {"", "unknown", "nan"} else lang[:2]
            best_values.append(zipf_frequency(str(row.word), lang, wordlist="best"))
        freq["frequency"] = best_values
        freq["log_frequency"] = best_values
        freq["wordfreq_zipf"] = best_values
        log("[prepare] aggregate frequency features from wordfreq")
    except Exception as exc:
        log(f"[warn] wordfreq unavailable, leaving NaN frequency values: {exc}")
    freq["word_key"] = freq["word"].astype(str).str.strip().str.lower()
    freq = freq.merge(_load_subtlex_us_counts(raw_dir), on="word_key", how="left")
    freq = freq.drop(columns=["word_key"])
    fallback_rank, fallback_percentile, fallback_bands = _rank_percentile_from_score(freq["wordfreq_zipf"])
    freq["frequency_rank"] = freq["subtlex_us_rank"].combine_first(fallback_rank)
    freq["frequency_rank_percentile"] = freq["subtlex_us_rank_percentile"].combine_first(fallback_percentile)
    frequency_bands = fallback_bands.copy()
    subtlex_rank = pd.to_numeric(freq["subtlex_us_rank"], errors="coerce")
    for cutoff in [1000, 2000, 3000, 5000, 10000, 20000]:
        selected = (subtlex_rank.notna() & (subtlex_rank <= cutoff)).to_numpy()
        frequency_bands[selected] = np.minimum(frequency_bands[selected], cutoff)
    freq["frequency_band"] = frequency_bands
    return ensure_frequency_columns(freq)


def ensure_frequency_columns(frequency: pd.DataFrame) -> pd.DataFrame:
    out = frequency.copy()
    for column in FREQUENCY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    if out["wordfreq_zipf"].isna().all() and "log_frequency" in out.columns:
        out["wordfreq_zipf"] = out["log_frequency"]
    if out["frequency_rank"].isna().all():
        values = out["wordfreq_zipf"].fillna(out["log_frequency"])
        rank, percentile, bands = _rank_percentile_from_score(values)
        out["frequency_rank"] = rank
        out["frequency_rank_percentile"] = percentile
        out["frequency_band"] = bands
    if out["frequency_band"].isna().any():
        rank_values = pd.to_numeric(out["frequency_rank"], errors="coerce")
        bands = np.full(len(out), 999999, dtype=np.int32)
        for cutoff in [1000, 2000, 3000, 5000, 10000, 20000]:
            selected = rank_values.notna() & (rank_values <= cutoff)
            bands[selected.to_numpy()] = np.minimum(bands[selected.to_numpy()], cutoff)
        out["frequency_band"] = out["frequency_band"].fillna(pd.Series(bands, index=out.index))
    return out[FREQUENCY_COLUMNS]


def add_l2_word_stats(frequency: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "Responses L2 English speakers to 62 thousand words.xlsx"
    out = frequency.copy()
    for column in ["accuracy", "acc_L2", "rank_L2", "nobs_L2", "acc_L1", "rank_L1", "diff_L1_L2"]:
        if column not in out.columns:
            out[column] = np.nan
    if not path.exists():
        log(f"[warn] L2 word stats file missing: {path}")
        return ensure_frequency_columns(out)
    stats = pd.read_excel(path, sheet_name="Words")
    if "spelling" not in stats.columns:
        raise ValueError(f"{path}: Words sheet missing spelling column")
    stats = stats.copy()
    stats["word_key"] = stats["spelling"].astype(str).str.strip().str.lower()
    rename_map = {
        "accuracy": "accuracy",
        "nobs": "nobs_L2",
        "rank_L2": "rank_L2",
        "acc_L1": "acc_L1",
        "rank_L1": "rank_L1",
        "diff_L1_L2": "diff_L1_L2",
    }
    cols = ["word_key"] + [c for c in rename_map if c in stats.columns]
    stats = stats[cols].rename(columns=rename_map)
    if "accuracy" in stats.columns:
        stats["acc_L2"] = stats["accuracy"]
    out["word_key"] = out["word"].astype(str).str.strip().str.lower()
    out = out.merge(stats, on="word_key", how="left", suffixes=("", "_from_l2"))
    for column in ["accuracy", "acc_L2", "rank_L2", "nobs_L2", "acc_L1", "rank_L1", "diff_L1_L2"]:
        src = f"{column}_from_l2"
        if src in out.columns:
            out[column] = out[src].combine_first(out[column])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_from_l2") or c == "word_key"])
    matched = int(out["accuracy"].notna().sum())
    log(f"[prepare] L2 word stats matched rows={matched}/{len(out)}")
    return ensure_frequency_columns(out)


def hash_embedding(word: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(word.lower().encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_hash_embeddings(words: pd.DataFrame, dim: int) -> tuple[np.ndarray, dict]:
    if words.empty:
        return np.zeros((0, dim), dtype=np.float32), {
            "embedding_backend": "hash",
            "dimension": dim,
            "empty": True,
        }
    embeddings = np.vstack([hash_embedding(str(word), dim) for word in words["word"].astype(str).tolist()]).astype(np.float32)
    return embeddings, {
        "embedding_backend": "hash",
        "dimension": dim,
        "normalized_l2": True,
        "aligned_to": "processed/words.csv row order",
        "warning": "Deterministic smoke-test embedding backend.",
    }


def _build_fasttext_vec_embeddings(words: pd.DataFrame, path_obj: Path, embedding_dim: int) -> tuple[np.ndarray, dict]:
    wanted_words = [str(word) for word in words["word"].tolist()]
    wanted: dict[str, list[int]] = {}
    for idx, word in enumerate(wanted_words):
        word_text = str(word)
        for key in {word_text, word_text.lower()}:
            wanted.setdefault(key, []).append(idx)
    matrix = np.zeros((len(wanted_words), embedding_dim), dtype=np.float32)
    found = np.zeros(len(wanted_words), dtype=bool)

    def consume_lines(lines: Iterable[bytes]) -> None:
        first = True
        for raw_line in lines:
            if first:
                first = False
                parts = raw_line.decode("utf-8", errors="ignore").strip().split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    continue
            parts = raw_line.decode("utf-8", errors="ignore").rstrip().split(" ")
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            if token not in wanted:
                continue
            vector = np.asarray(parts[1:], dtype=np.float32)
            for idx in wanted[token]:
                if not found[idx]:
                    matrix[idx] = vector
                    found[idx] = True

    if path_obj.suffix == ".zip":
        with zipfile.ZipFile(path_obj) as archive:
            vec_names = [name for name in archive.namelist() if name.endswith(".vec")]
            if len(vec_names) != 1:
                raise ValueError(f"{path_obj}: expected exactly one .vec file, found {vec_names}")
            with archive.open(vec_names[0]) as handle:
                consume_lines(handle)
    else:
        with path_obj.open("rb") as handle:
            consume_lines(handle)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, np.maximum(norms, 1e-12), out=np.zeros_like(matrix))
    return matrix, {
        "embedding_backend": "fasttext",
        "fasttext_format": "vec",
        "dimension": int(matrix.shape[1]),
        "model_path": str(path_obj),
        "normalized_l2": True,
        "aligned_to": "processed/words.csv row order",
        "oov_count": int((~found).sum()),
        "oov_fraction": float((~found).mean()) if len(found) > 0 else 0.0,
    }


def build_fasttext_embeddings(words: pd.DataFrame, lang: str, model_path: str | None, download_fasttext: bool, embedding_dim: int) -> tuple[np.ndarray, dict]:
    import fasttext
    import fasttext.util
    language = str(lang).strip().lower()[:2]
    chosen_path = model_path
    if chosen_path is None:
        if not download_fasttext:
            raise RuntimeError("fastText model path missing and download disabled")
        log(f"[download] fastText model cc.{language}.300.bin")
        fasttext.util.download_model(language, if_exists="ignore")
        chosen_path = f"cc.{language}.300.bin"
    path_obj = Path(chosen_path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    if path_obj.suffix in {".vec", ".zip"}:
        embeddings, metadata = _build_fasttext_vec_embeddings(words, path_obj, embedding_dim)
        metadata["language"] = language
        return embeddings, metadata
    model = fasttext.load_model(str(path_obj))
    original_dim = model.get_dimension()
    if embedding_dim != original_dim:
        if embedding_dim < 1 or embedding_dim > original_dim:
            raise ValueError(f"invalid embedding_dim={embedding_dim} for fastText dimension={original_dim}")
        fasttext.util.reduce_model(model, embedding_dim)
    matrix = np.vstack([model.get_word_vector(str(word)) for word in words["word"].astype(str).tolist()]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, np.maximum(norms, 1e-12), out=np.zeros_like(matrix))
    return matrix, {
        "embedding_backend": "fasttext",
        "dimension": int(matrix.shape[1]),
        "language": language,
        "model_path": str(path_obj),
        "original_dimension": int(original_dim),
        "normalized_l2": True,
        "aligned_to": "processed/words.csv row order",
    }


def build_words_table(static_responses: pd.DataFrame, temporal_responses: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not static_responses.empty:
        frames.append(
            static_responses[["word_id", "word", "language"]]
            .assign(lemma=pd.NA, pos=pd.NA, morphology=pd.NA, source="static")
        )
    if not temporal_responses.empty:
        frames.append(
            temporal_responses[["word_id", "word", "lemma", "pos", "morphology", "learning_language"]]
            .rename(columns={"learning_language": "language"})
            .assign(source="duolingo_hlr")
        )
    if not frames:
        return pd.DataFrame(columns=WORDS_COLUMNS)
    words = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["word_id"], keep="first")
    words["word"] = words["word"].astype(str)
    words["length"] = words["word"].str.len()
    return words[WORDS_COLUMNS]


def create_splits(static_responses: pd.DataFrame, seed: int) -> tuple[dict, dict, dict]:
    users = sorted(static_responses["user_id"].astype(str).unique().tolist())
    word_ids = sorted(static_responses["word_id"].astype(str).unique().tolist())
    loo = {
        "description": "Leave-one-user-out splits for static vocabulary benchmark.",
        "splits": [{"test_user": user, "train_users": [u for u in users if u != user]} for user in users],
    }
    rng = np.random.default_rng(seed)
    users_shuffled = users.copy()
    rng.shuffle(users_shuffled)
    n_val = max(1, int(round(0.2 * len(users_shuffled)))) if users_shuffled else 0
    val = {
        "description": "Validation/test user split.",
        "validation_users": users_shuffled[:n_val],
        "test_users": users_shuffled[n_val:],
    }
    words_shuffled = word_ids.copy()
    rng.shuffle(words_shuffled)
    n_cold = max(1, int(round(0.1 * len(words_shuffled)))) if words_shuffled else 0
    cold = {
        "description": "Cold-word split. Hold out these words globally during training.",
        "cold_word_ids": words_shuffled[:n_cold],
        "train_word_ids": words_shuffled[n_cold:],
    }
    return loo, val, cold


def write_dataset_card(paths: Paths, data_mode: str) -> None:
    content = {
        "generated_by": "scripts/prepare_data.py",
        "data_mode": data_mode,
        "files": {
            "processed/responses_static.csv": "Primary static known/unknown labels per user-word.",
            "processed/responses_temporal.csv": "Secondary temporal recall traces (Duolingo-style).",
            "processed/words.csv": "Word inventory aligned with embeddings rows.",
            "processed/frequency.csv": "Backed frequency features from wordfreq and SUBTLEXus, plus L2 word statistics when available.",
            "processed/embeddings.npy": "Embedding matrix aligned to words.csv row order.",
            "processed/embeddings_metadata.json": "Embedding backend metadata.",
        },
        "manual_data_instructions": {
            "ehara": "Place CSV at data/raw/ehara_esl_vocab/responses_raw.csv or pass --ehara-raw.",
            "evkd1": "Place CSV at data/raw/evkd1/responses_raw.csv or pass --evkd1-raw.",
            "duolingo_hlr": "Place CSV(.gz) at data/raw/duolingo_hlr/learning_traces.csv.gz or pass --duolingo-raw.",
        },
    }
    (paths.data_dir / "DATASET_CARD.json").write_text(json.dumps(content, indent=2), encoding="utf-8")


def write_manual_download_notes(paths: Paths) -> None:
    note = (
        "Manual dataset download notes\n\n"
        "Ehara ESL: save static CSV to data/raw/ehara_esl_vocab/responses_raw.csv\n"
        "EVKD1: save static CSV to data/raw/evkd1/responses_raw.csv\n"
        "Duolingo HLR: save traces CSV(.gz) to data/raw/duolingo_hlr/learning_traces.csv.gz\n"
    )
    (paths.raw_dir / "MANUAL_DOWNLOAD_INSTRUCTIONS.txt").write_text(note, encoding="utf-8")


def resolve_input_path(explicit_path: str | None, default_path: Path) -> Path | None:
    if explicit_path is not None:
        source = Path(explicit_path)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.resolve() != default_path.resolve():
            default_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, default_path)
            log(f"[copy] {source} -> {default_path}")
        return default_path
    return default_path if default_path.exists() else None


def validate_outputs(paths: Paths) -> None:
    required_paths = [
        paths.processed_dir / "responses_static.csv",
        paths.processed_dir / "responses_temporal.csv",
        paths.processed_dir / "words.csv",
        paths.processed_dir / "frequency.csv",
        paths.processed_dir / "embeddings.npy",
        paths.processed_dir / "embeddings_metadata.json",
        paths.data_dir / "DATASET_CARD.json",
    ]
    for path in required_paths:
        if not path.exists():
            raise ValueError(f"missing required file: {path}")
    static_df = pd.read_csv(paths.processed_dir / "responses_static.csv")
    temporal_df = pd.read_csv(paths.processed_dir / "responses_temporal.csv")
    words_df = pd.read_csv(paths.processed_dir / "words.csv")
    frequency_df = pd.read_csv(paths.processed_dir / "frequency.csv")
    embeddings = np.load(paths.processed_dir / "embeddings.npy")
    if list(static_df.columns) != STATIC_COLUMNS:
        raise ValueError("responses_static.csv columns mismatch")
    if list(temporal_df.columns) != TEMPORAL_COLUMNS:
        raise ValueError("responses_temporal.csv columns mismatch")
    if list(words_df.columns) != WORDS_COLUMNS:
        raise ValueError("words.csv columns mismatch")
    if list(frequency_df.columns) != FREQUENCY_COLUMNS:
        raise ValueError("frequency.csv columns mismatch")
    if len(words_df) != embeddings.shape[0]:
        raise ValueError(f"embedding alignment mismatch len(words)={len(words_df)} embeddings_rows={embeddings.shape[0]}")
    if not static_df.empty:
        if not static_df["label"].isin([0, 1]).all():
            raise ValueError("responses_static labels are not binary")
        word_ids = set(words_df["word_id"].astype(str).tolist())
        response_word_ids = set(static_df["word_id"].astype(str).tolist())
        if not response_word_ids.issubset(word_ids):
            raise ValueError("responses_static has unknown word_id")
        loo = json.loads((paths.splits_dir / "static_leave_one_user_out.json").read_text(encoding="utf-8"))
        val = json.loads((paths.splits_dir / "static_validation_users.json").read_text(encoding="utf-8"))
        cold = json.loads((paths.splits_dir / "cold_word_split.json").read_text(encoding="utf-8"))
        users = set(static_df["user_id"].astype(str).tolist())
        for row in loo["splits"]:
            if str(row["test_user"]) not in users:
                raise ValueError("split references unknown test_user")
        for user_id in val["validation_users"] + val["test_users"]:
            if str(user_id) not in users:
                raise ValueError("validation split references unknown user")
        if not set(map(str, cold["cold_word_ids"] + cold["train_word_ids"])).issubset(set(words_df["word_id"].astype(str).tolist())):
            raise ValueError("cold word split references unknown words")
    log("[validate] validation passed")


def main() -> None:
    log("[start] prepare_data.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ehara-raw", default=None)
    parser.add_argument("--evkd1-raw", default=None)
    parser.add_argument("--duolingo-raw", default=None)
    parser.add_argument("--duolingo-max-rows", type=int, default=None)
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--static-binarization", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--embedding-backend", choices=["hash", "fasttext"], default="hash")
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--fasttext-model-path", default=None)
    parser.add_argument("--download-fasttext", action="store_true")
    parser.add_argument("--fasttext-lang", default="en")
    parser.add_argument("--allow-hash-fallback", action="store_true")
    parser.add_argument("--synthetic-if-missing", action="store_true")
    parser.add_argument("--synthetic-users", type=int, default=100)
    parser.add_argument("--synthetic-words", type=int, default=1000)
    parser.add_argument("--synthetic-embedding-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    log(f"[args] {args}")

    paths = ensure_paths(Path(args.data_dir))
    log(f"[paths] data_dir={paths.data_dir.resolve()}")
    if args.skip_downloads:
        log("[mode] downloads skipped")
    else:
        log("[mode] downloads enabled for metadata/resources only")
        duolingo_readme_url = "https://raw.githubusercontent.com/duolingo/halflife-regression/master/README.md"
        dest = paths.raw_duolingo_dir / "README.md"
        try:
            with urllib.request.urlopen(duolingo_readme_url, timeout=30) as response:
                dest.write_bytes(response.read())
            log(f"[download] {duolingo_readme_url} -> {dest}")
        except Exception as exc:
            log(f"[warn] could not download Duolingo README: {exc}")
    write_manual_download_notes(paths)

    ehara = resolve_input_path(args.ehara_raw, paths.raw_ehara_dir / "responses_raw.csv")
    evkd1 = resolve_input_path(args.evkd1_raw, paths.raw_evkd1_dir / "responses_raw.csv")
    duolingo = resolve_input_path(args.duolingo_raw, paths.raw_duolingo_dir / "learning_traces.csv.gz")
    log(f"[source] ehara={'found' if ehara is not None else 'missing'}")
    log(f"[source] evkd1={'found' if evkd1 is not None else 'missing'}")
    log(f"[source] duolingo={'found' if duolingo is not None else 'missing'}")

    static_frames: list[pd.DataFrame] = []
    if ehara is not None:
        static_frames.append(normalize_static_dataset(ehara, "ehara_esl_vocab", args.static_binarization))
    if evkd1 is not None:
        static_frames.append(normalize_static_dataset(evkd1, "evkd1", args.static_binarization))
    temporal_df = normalize_duolingo_dataset(duolingo, args.duolingo_max_rows) if duolingo is not None else pd.DataFrame(columns=TEMPORAL_COLUMNS)

    if static_frames:
        static_df = pd.concat(static_frames, ignore_index=True).drop_duplicates(subset=["source", "user_id", "word_id"], keep="first")
        words_df = build_words_table(static_df, temporal_df)
        frequency_df = add_frequency_features(words_df, paths.raw_dir)
        frequency_df = add_l2_word_stats(frequency_df, paths.raw_dir)
        if args.embedding_backend == "fasttext":
            try:
                embeddings, emb_meta = build_fasttext_embeddings(words_df, args.fasttext_lang, args.fasttext_model_path, args.download_fasttext, args.embedding_dim)
            except Exception as exc:
                if not args.allow_hash_fallback:
                    raise RuntimeError(
                        "fastText embedding backend requested but fastText embeddings could not be built. "
                        "Provide --fasttext-model-path, use --download-fasttext, or pass --allow-hash-fallback "
                        f"only for smoke tests. Root cause: {type(exc).__name__}: {exc}"
                    ) from exc
                log(f"[warn] fastText failed, fallback to hash: {exc}")
                embeddings, emb_meta = build_hash_embeddings(words_df, args.embedding_dim)
                emb_meta["fallback_reason"] = str(exc)
        else:
            embeddings, emb_meta = build_hash_embeddings(words_df, args.embedding_dim)
        data_mode = "real"
    elif args.synthetic_if_missing:
        static_df, words_df, frequency_df, embeddings, emb_meta = generate_synthetic_static_dataset(
            n_users=args.synthetic_users,
            n_words=args.synthetic_words,
            embedding_dim=args.synthetic_embedding_dim,
            seed=args.seed,
        )
        frequency_df = ensure_frequency_columns(frequency_df)
        data_mode = "synthetic"
    else:
        static_df = pd.DataFrame(columns=STATIC_COLUMNS)
        words_df = build_words_table(static_df, temporal_df)
        frequency_df = add_frequency_features(words_df, paths.raw_dir)
        frequency_df = add_l2_word_stats(frequency_df, paths.raw_dir)
        embeddings, emb_meta = build_hash_embeddings(words_df, args.embedding_dim)
        emb_meta["empty"] = True
        data_mode = "placeholder"

    static_path = paths.processed_dir / "responses_static.csv"
    temporal_path = paths.processed_dir / "responses_temporal.csv"
    words_path = paths.processed_dir / "words.csv"
    freq_path = paths.processed_dir / "frequency.csv"
    emb_path = paths.processed_dir / "embeddings.npy"
    emb_meta_path = paths.processed_dir / "embeddings_metadata.json"
    static_df[STATIC_COLUMNS].to_csv(static_path, index=False)
    temporal_df[TEMPORAL_COLUMNS].to_csv(temporal_path, index=False)
    words_df[WORDS_COLUMNS].to_csv(words_path, index=False)
    frequency_df[FREQUENCY_COLUMNS].to_csv(freq_path, index=False)
    np.save(emb_path, embeddings.astype(np.float32))
    emb_meta["aligned_to"] = "processed/words.csv row order"
    emb_meta["data_mode"] = data_mode
    emb_meta_path.write_text(json.dumps(emb_meta, indent=2), encoding="utf-8")
    log(f"[write] {static_path} rows={len(static_df)}")
    log(f"[write] {temporal_path} rows={len(temporal_df)}")
    log(f"[write] {words_path} rows={len(words_df)}")
    log(f"[write] {freq_path} rows={len(frequency_df)}")
    log(f"[write] {emb_path} shape={embeddings.shape}")
    log(f"[write] {emb_meta_path}")

    if not static_df.empty:
        loo, val, cold = create_splits(static_df, seed=args.seed)
        (paths.splits_dir / "static_leave_one_user_out.json").write_text(json.dumps(loo, indent=2), encoding="utf-8")
        (paths.splits_dir / "static_validation_users.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        (paths.splits_dir / "cold_word_split.json").write_text(json.dumps(cold, indent=2), encoding="utf-8")
        log(f"[write] {paths.splits_dir / 'static_leave_one_user_out.json'}")
        log(f"[write] {paths.splits_dir / 'static_validation_users.json'}")
        log(f"[write] {paths.splits_dir / 'cold_word_split.json'}")
    else:
        for name in ["static_leave_one_user_out.json", "static_validation_users.json", "cold_word_split.json"]:
            path = paths.splits_dir / name
            path.write_text(json.dumps({"description": "No static data available", "splits": []}, indent=2), encoding="utf-8")
            log(f"[write] {path}")

    write_dataset_card(paths, data_mode)
    log(f"[write] {paths.data_dir / 'DATASET_CARD.json'}")

    validate_outputs(paths)
    known_fraction = float(static_df["label"].mean()) if not static_df.empty else float("nan")
    log(
        "[summary] "
        f"mode={data_mode} users={static_df['user_id'].nunique() if not static_df.empty else 0} "
        f"words={len(words_df)} labels={len(static_df)} known_fraction={known_fraction} "
        f"embedding_shape={embeddings.shape}"
    )
    log("[done] data preparation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        raise SystemExit(1) from exc
