from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_STATIC_COLUMNS: tuple[str, ...] = (
    "user_id",
    "word_id",
    "word",
    "label",
    "raw_score",
    "timestamp",
    "source",
    "language",
)


@dataclass(frozen=True)
class LoadedData:
    words: pd.DataFrame
    frequency: pd.DataFrame
    embeddings: np.ndarray
    responses_static: pd.DataFrame
    responses_temporal: pd.DataFrame
    splits: dict[str, Any]
    embedding_metadata: dict[str, Any]
    embedding_backend: str


def load_words(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed" / "words.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_frequency(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed" / "frequency.csv"
    if not path.exists():
        return pd.DataFrame(columns=["word_id", "word", "language", "frequency", "log_frequency"])
    return pd.read_csv(path)


def load_embeddings(data_dir: Path) -> np.ndarray:
    path = data_dir / "processed" / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def load_static_responses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed" / "responses_static.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(REQUIRED_STATIC_COLUMNS))
    return pd.read_csv(path)


def load_temporal_responses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed" / "responses_temporal.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_splits(data_dir: Path) -> dict[str, Any]:
    split_dir = data_dir / "splits"
    out: dict[str, Any] = {}
    for name in ["static_leave_one_user_out", "static_validation_users", "cold_word_split"]:
        path = split_dir / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _load_embedding_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "processed" / "embeddings_metadata.json"
    if not path.exists():
        return {"embedding_backend": "unknown"}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_loaded_data(words: pd.DataFrame, embeddings: np.ndarray, responses_static: pd.DataFrame, splits: dict[str, Any]) -> None:
    if len(words) != embeddings.shape[0]:
        raise ValueError(f"words/embedding mismatch: {len(words)} vs {embeddings.shape[0]}")
    if not responses_static.empty:
        missing_cols = [c for c in REQUIRED_STATIC_COLUMNS if c not in responses_static.columns]
        if missing_cols:
            raise ValueError(f"responses_static missing columns: {missing_cols}")
        if not responses_static["label"].isin([0, 1]).all():
            raise ValueError("responses_static labels must be binary 0/1")
        word_ids = set(words["word_id"].astype(str).tolist())
        response_word_ids = set(responses_static["word_id"].astype(str).tolist())
        if not response_word_ids.issubset(word_ids):
            raise ValueError("responses_static contains unknown word_id")
        dup = responses_static.duplicated(subset=["source", "user_id", "word_id"], keep=False)
        if dup.any():
            raise ValueError("duplicate static rows for source,user_id,word_id")
    if "static_leave_one_user_out" in splits and not responses_static.empty:
        users = set(responses_static["user_id"].astype(str).tolist())
        payload = splits["static_leave_one_user_out"]
        split_rows = payload.get("splits", payload if isinstance(payload, list) else [])
        split_users = set()
        for row in split_rows:
            test_user = row.get("test_user_id", row.get("test_user"))
            if test_user is not None:
                split_users.add(str(test_user))
        if not split_users.issubset(users):
            raise ValueError("split user ids not found in responses_static")
    if "cold_word_split" in splits and not words.empty:
        word_ids = set(words["word_id"].astype(str).tolist())
        cold_ids = set(map(str, splits["cold_word_split"].get("cold_word_ids", [])))
        if not cold_ids.issubset(word_ids):
            raise ValueError("cold_word_split includes unknown word ids")


def load_all(data_dir: Path) -> LoadedData:
    words = load_words(data_dir)
    frequency = load_frequency(data_dir)
    embeddings = load_embeddings(data_dir)
    responses_static = load_static_responses(data_dir)
    responses_temporal = load_temporal_responses(data_dir)
    splits = load_splits(data_dir)
    metadata = _load_embedding_metadata(data_dir)
    backend = str(metadata.get("embedding_backend", "unknown"))
    validate_loaded_data(words, embeddings, responses_static, splits)
    return LoadedData(
        words=words,
        frequency=frequency,
        embeddings=embeddings,
        responses_static=responses_static,
        responses_temporal=responses_temporal,
        splits=splits,
        embedding_metadata=metadata,
        embedding_backend=backend,
    )
