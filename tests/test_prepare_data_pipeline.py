from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_prepare(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-u", "scripts/prepare_data.py"] + args
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def test_smoke_synthetic_creates_files(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "data"
    result = run_prepare(
        [
            "--data-dir",
            str(out_dir),
            "--embedding-backend",
            "hash",
            "--skip-downloads",
            "--synthetic-if-missing",
            "--synthetic-users",
            "60",
            "--synthetic-words",
            "400",
            "--synthetic-embedding-dim",
            "40",
            "--seed",
            "7",
        ],
        repo,
    )
    assert "[validate] validation passed" in result.stdout
    required = [
        out_dir / "processed" / "responses_static.csv",
        out_dir / "processed" / "responses_temporal.csv",
        out_dir / "processed" / "words.csv",
        out_dir / "processed" / "frequency.csv",
        out_dir / "processed" / "embeddings.npy",
        out_dir / "processed" / "embeddings_metadata.json",
        out_dir / "splits" / "static_leave_one_user_out.json",
        out_dir / "splits" / "static_validation_users.json",
        out_dir / "splits" / "cold_word_split.json",
        out_dir / "DATASET_CARD.json",
    ]
    for path in required:
        assert path.exists()
    static_df = pd.read_csv(out_dir / "processed" / "responses_static.csv")
    words_df = pd.read_csv(out_dir / "processed" / "words.csv")
    embeddings = np.load(out_dir / "processed" / "embeddings.npy")
    assert len(static_df) > 0
    assert static_df["label"].isin([0, 1]).all()
    assert len(words_df) == embeddings.shape[0]


def test_static_normalization_ordinal_binarization(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "ehara_esl_vocab"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "learner_id": ["u1", "u1", "u2", "u2"],
            "surface_form": ["apple", "banana", "apple", "banana"],
            "rating": [5, 3, 4, 5],
        }
    )
    raw_path = raw_dir / "responses_raw.csv"
    raw.to_csv(raw_path, index=False)
    run_prepare(
        [
            "--data-dir",
            str(data_dir),
            "--embedding-backend",
            "hash",
            "--skip-downloads",
            "--ehara-raw",
            str(raw_path),
            "--static-binarization",
            "strict",
        ],
        repo,
    )
    strict_df = pd.read_csv(data_dir / "processed" / "responses_static.csv")
    assert strict_df["label"].sum() == 2
    run_prepare(
        [
            "--data-dir",
            str(data_dir),
            "--embedding-backend",
            "hash",
            "--skip-downloads",
            "--ehara-raw",
            str(raw_path),
            "--static-binarization",
            "relaxed",
        ],
        repo,
    )
    relaxed_df = pd.read_csv(data_dir / "processed" / "responses_static.csv")
    assert relaxed_df["label"].sum() >= strict_df["label"].sum()


def test_hash_embedding_deterministic(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "ehara_esl_vocab"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "word": ["cat", "dog", "cat", "dog"],
            "label": [1, 0, 1, 1],
        }
    )
    raw_path = raw_dir / "responses_raw.csv"
    raw.to_csv(raw_path, index=False)
    args = [
        "--data-dir",
        str(data_dir),
        "--embedding-backend",
        "hash",
        "--skip-downloads",
        "--ehara-raw",
        str(raw_path),
    ]
    run_prepare(args, repo)
    emb1 = np.load(data_dir / "processed" / "embeddings.npy")
    run_prepare(args, repo)
    emb2 = np.load(data_dir / "processed" / "embeddings.npy")
    assert np.allclose(emb1, emb2)
    splits = json.loads((data_dir / "splits" / "static_leave_one_user_out.json").read_text(encoding="utf-8"))
    assert "splits" in splits and len(splits["splits"]) == 2
