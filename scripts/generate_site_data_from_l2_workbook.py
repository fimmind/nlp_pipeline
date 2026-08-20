#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from vocab_benchmark.estimators.irt import Response12GroupedResidualIRTEstimator


WORDS_SHEET = "Words"
WORD_COLUMN = "spelling"
ACCURACY_COLUMN = "accuracy"
GROUP_COUNT = 12


def _normalize_word(word: Any) -> str:
    text = str(word)
    if text == "nan":
        return ""
    return text.strip().lower().replace("’", "'")


def _stable_word_id(word: str) -> str:
    digest = hashlib.sha1(word.encode("utf-8")).hexdigest()[:16]
    return f"w_{digest}"


def _normalize_colname(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_colname(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_colname(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _coerce_binary_label(series: pd.Series, raw_score: pd.Series | None, binarization: str) -> pd.Series:
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


def _normalize_static_dataset(raw_path: Path, source: str, binarization: str) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"raw static file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    user_col = _find_col(df, ["user_id", "userid", "user", "learner_id", "learner", "student_id"])
    word_col = _find_col(df, ["word", "surface", "surface_form", "vocabulary", "item", "token"])
    label_col = _find_col(df, ["label", "known", "is_known", "answer", "correct", "binary_label"])
    score_col = _find_col(df, ["raw_score", "score", "rating", "knowledge", "knowledge_score"])

    if user_col is None:
        raise ValueError(f"{raw_path}: cannot infer user_id column")
    if word_col is None:
        raise ValueError(f"{raw_path}: cannot infer word column")
    if label_col is None and score_col is None:
        raise ValueError(f"{raw_path}: cannot infer label/raw_score column")

    out = pd.DataFrame()
    out["user_id"] = df[user_col].astype(str)
    out["word"] = df[word_col].astype(str)

    raw_score = df[score_col] if score_col is not None else None
    if label_col is not None:
        out["label"] = _coerce_binary_label(df[label_col], raw_score, binarization)
    else:
        assert raw_score is not None
        out["label"] = _coerce_binary_label(raw_score, raw_score, binarization)

    out["source"] = source
    out = out.dropna(subset=["user_id", "word", "label"]).copy()
    out["label"] = out["label"].astype(np.int32)
    out = out[(out["label"] == 0) | (out["label"] == 1)]
    out = out.drop_duplicates(subset=["source", "user_id", "word"], keep="first")
    return out[["source", "user_id", "word", "label"]]


def _load_words_and_accuracy(workbook_path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"workbook not found: {workbook_path}")

    sheet = pd.read_excel(workbook_path, sheet_name=WORDS_SHEET)
    if WORD_COLUMN not in sheet.columns:
        raise ValueError(f"missing required column in {WORDS_SHEET}: {WORD_COLUMN}")
    if ACCURACY_COLUMN not in sheet.columns:
        raise ValueError(f"missing required column in {WORDS_SHEET}: {ACCURACY_COLUMN}")

    raw_words = sheet[WORD_COLUMN].astype(str)
    raw_accuracy = pd.to_numeric(sheet[ACCURACY_COLUMN], errors="coerce")

    dedup: OrderedDict[str, tuple[float, int]] = OrderedDict()
    dropped_empty = 0

    for word_raw, acc_raw in zip(raw_words.tolist(), raw_accuracy.tolist()):
        word_norm = _normalize_word(word_raw)
        if word_norm == "" or word_norm == "nan":
            dropped_empty += 1
            continue

        if acc_raw is None or (isinstance(acc_raw, float) and np.isnan(acc_raw)):
            accuracy = 0.5
        else:
            accuracy = float(acc_raw)
        if accuracy > 1.0:
            accuracy = accuracy / 100.0
        accuracy = float(np.clip(accuracy, 1e-6, 1.0 - 1e-6))

        if word_norm in dedup:
            total, count = dedup[word_norm]
            dedup[word_norm] = (total + accuracy, count + 1)
        else:
            dedup[word_norm] = (accuracy, 1)

    words: list[str] = []
    acc_values: list[float] = []
    duplicate_rows = 0

    for word, (acc_sum, count) in dedup.items():
        words.append(word)
        acc_values.append(float(acc_sum / float(count)))
        if count > 1:
            duplicate_rows += count - 1

    if len(words) == 0:
        raise ValueError("no words were extracted from workbook")

    metadata: dict[str, Any] = {
        "source_sheet": WORDS_SHEET,
        "source_word_column": WORD_COLUMN,
        "source_accuracy_column": ACCURACY_COLUMN,
        "dropped_empty_rows": int(dropped_empty),
        "duplicate_rows_collapsed": int(duplicate_rows),
        "final_word_count": int(len(words)),
    }
    return words, np.asarray(acc_values, dtype=np.float64), metadata


def _build_words_df(words: list[str]) -> pd.DataFrame:
    word_idx = np.arange(len(words), dtype=np.int32)
    word_id = [_stable_word_id(word) for word in words]
    return pd.DataFrame({"word_idx": word_idx, "word_id": word_id, "word": words})


def _build_difficulties_df(words_df: pd.DataFrame, accuracy: np.ndarray) -> pd.DataFrame:
    if len(words_df) != len(accuracy):
        raise ValueError(f"words/accuracy length mismatch: {len(words_df)} vs {len(accuracy)}")
    out = words_df.copy()
    out["accuracy"] = accuracy.astype(np.float64)
    return out[["word_idx", "word_id", "word", "accuracy"]]


def _load_raw_static_responses(
    ehara_raw: Path,
    evkd1_raw: Path,
    binarization: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sources_used: list[str] = []

    if ehara_raw.exists():
        frames.append(_normalize_static_dataset(ehara_raw, "ehara_esl_vocab", binarization))
        sources_used.append("ehara_esl_vocab")
    if evkd1_raw.exists():
        frames.append(_normalize_static_dataset(evkd1_raw, "evkd1", binarization))
        sources_used.append("evkd1")

    if len(frames) == 0:
        raise FileNotFoundError(
            "No raw static response files found. Expected at least one of: "
            f"{ehara_raw} or {evkd1_raw}"
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["source", "user_id", "word"], keep="first")
    metadata = {
        "raw_static_sources_used": sources_used,
        "raw_static_rows": int(len(out)),
    }
    return out, metadata


def _build_response_frame(
    words_df: pd.DataFrame,
    raw_static: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    words_df_local = words_df.copy()
    words_df_local["word_norm"] = words_df_local["word"].map(_normalize_word)
    word_index_by_norm = {
        word: int(idx) for word, idx in zip(words_df_local["word_norm"].tolist(), words_df_local["word_idx"].tolist())
    }

    frame = raw_static.copy()
    frame["word_norm"] = frame["word"].map(_normalize_word)
    frame["word_idx"] = frame["word_norm"].map(word_index_by_norm)
    frame = frame.dropna(subset=["word_idx"]).copy()
    frame["word_idx"] = frame["word_idx"].astype(np.int32)
    frame["label"] = frame["label"].astype(np.int32)

    frame["user_key"] = frame["source"].astype(str) + ":" + frame["user_id"].astype(str)
    users = sorted(frame["user_key"].unique().tolist())
    user_index = {user_key: idx for idx, user_key in enumerate(users)}
    frame["user_idx"] = frame["user_key"].map(user_index).astype(np.int32)

    out = frame[["user_idx", "word_idx", "label", "user_key"]].copy()
    out = out.rename(columns={"user_key": "user_id"})

    metadata = {
        "response_rows_used": int(len(out)),
        "response_users_used": int(len(users)),
        "response_words_mapped": int(out["word_idx"].nunique()),
    }
    return out, metadata


def _build_group_matrix_from_response_patterns(
    response_frame: pd.DataFrame,
    accuracy: np.ndarray,
    n_words: int,
    seed: int,
) -> np.ndarray:
    dummy_features = np.zeros((n_words, 1), dtype=np.float32)
    estimator = Response12GroupedResidualIRTEstimator(
        tau_theta=2.0,
        tau_delta=1.6,
        gate_c=12.0,
        n_groups=GROUP_COUNT,
        random_state=seed,
        threshold_min=0.10,
        threshold_max=0.90,
        threshold_step=0.005,
        threshold_shrink_c=30.0,
        accuracy_values=accuracy.astype(np.float64),
        use_accuracy_difficulty=True,
    )
    estimator.fit(train_responses=response_frame, word_features=dummy_features)
    return estimator.q_matrix.astype(np.float32)


def _build_group_df(words_df: pd.DataFrame, q: np.ndarray, n_groups: int, seed: int) -> pd.DataFrame:
    if q.shape[0] != len(words_df) or q.shape[1] != n_groups:
        raise ValueError(f"group matrix shape mismatch: {q.shape} vs ({len(words_df)}, {n_groups})")
    col_names = [f"q_{idx:02d}" for idx in range(n_groups)]
    out = pd.DataFrame(q, columns=col_names)
    out.insert(0, "word", words_df["word"].astype(str).to_numpy())
    out.insert(0, "word_id", words_df["word_id"].astype(str).to_numpy())
    out.insert(0, "word_idx", words_df["word_idx"].to_numpy(dtype=np.int32))
    out.insert(0, "seed", int(seed))
    out.insert(0, "n_groups", int(n_groups))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate site data from L2 workbook words/accuracy and raw-static response-pattern groupings."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("data/raw/Responses L2 English speakers to 62 thousand words.xlsx"),
        help="Path to workbook with Words sheet and spelling/accuracy columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/site_data"),
        help="Output directory for generated site data.",
    )
    parser.add_argument(
        "--ehara-raw",
        type=Path,
        default=Path("data/raw/ehara_esl_vocab/responses_raw.csv"),
        help="Path to raw Ehara static responses CSV.",
    )
    parser.add_argument(
        "--evkd1-raw",
        type=Path,
        default=Path("data/raw/evkd1/responses_raw.csv"),
        help="Path to raw EVKD1 static responses CSV (optional if missing).",
    )
    parser.add_argument(
        "--static-binarization",
        choices=["strict", "relaxed"],
        default="strict",
        help="Binarization mode for raw static scores.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic KMeans grouping.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    words, accuracy, load_meta = _load_words_and_accuracy(args.workbook)
    words_df = _build_words_df(words)
    difficulties_df = _build_difficulties_df(words_df, accuracy)

    raw_static, raw_meta = _load_raw_static_responses(
        ehara_raw=args.ehara_raw,
        evkd1_raw=args.evkd1_raw,
        binarization=args.static_binarization,
    )
    response_frame, response_meta = _build_response_frame(words_df=words_df, raw_static=raw_static)

    words_path = out_dir / "words.csv"
    difficulties_path = out_dir / "difficulties.csv"
    words_df.to_csv(words_path, index=False)
    difficulties_df.to_csv(difficulties_path, index=False)

    q = _build_group_matrix_from_response_patterns(
        response_frame=response_frame,
        accuracy=accuracy,
        n_words=len(words_df),
        seed=args.seed,
    )
    group_df = _build_group_df(words_df=words_df, q=q, n_groups=GROUP_COUNT, seed=args.seed)
    group_path = out_dir / f"grouped_residual_q_g{GROUP_COUNT}_seed{args.seed}.csv"
    group_df.to_csv(group_path, index=False)

    row_sums = q.sum(axis=1)
    group_output = {
        "n_groups": int(GROUP_COUNT),
        "path": str(group_path),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }

    metadata = {
        "source_workbook": str(args.workbook),
        "source_sheet": WORDS_SHEET,
        "generator": "scripts/generate_site_data_from_l2_workbook.py",
        "seed": int(args.seed),
        "groups": [GROUP_COUNT],
        "words_path": str(words_path),
        "difficulties_path": str(difficulties_path),
        "load_metadata": load_meta,
        "raw_static_metadata": raw_meta,
        "response_metadata": response_meta,
        "group_output": group_output,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote {words_path} rows={len(words_df)}")
    print(f"wrote {difficulties_path} rows={len(difficulties_df)}")
    print(
        "wrote {path} q_min={q_min:.8f} q_max={q_max:.8f} row_sum_min={row_sum_min:.8f} row_sum_max={row_sum_max:.8f}".format(
            **group_output
        )
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
