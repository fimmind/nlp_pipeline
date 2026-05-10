from __future__ import annotations

import numpy as np
import pandas as pd


FREQUENCY_FEATURE_COLUMNS: list[str] = [
    "wordfreq_zipf",
    "log_frequency_subtlex_us",
    "subtlex_us_count",
    "subtlex_us_rank_percentile",
    "frequency_rank_percentile",
]
L2_FEATURE_COLUMNS: list[str] = [
    "accuracy",
    "acc_L2",
    "rank_L2",
    "nobs_L2",
    "acc_L1",
    "rank_L1",
    "diff_L1_L2",
]
FREQUENCY_BANDS: list[int] = [1000, 2000, 3000, 5000, 10000, 20000, 999999]


def build_word_index(words: pd.DataFrame) -> dict[str, int]:
    return {str(word_id): idx for idx, word_id in enumerate(words["word_id"].astype(str).tolist())}


def _normalized_numeric_column(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.zeros((len(values), 1), dtype=np.float32)
    filled = numeric.fillna(numeric.median()).to_numpy(dtype=np.float32)
    mean = float(np.mean(filled))
    std = float(np.std(filled))
    if std <= 1e-8:
        return np.zeros((len(values), 1), dtype=np.float32)
    return ((filled - mean) / std).reshape(-1, 1).astype(np.float32)


def _raw_numeric_column(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.zeros((len(values), 1), dtype=np.float32)
    return numeric.fillna(numeric.median()).to_numpy(dtype=np.float32).reshape(-1, 1)


def _frequency_band_matrix(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(999999).to_numpy(dtype=np.int32)
    out = np.zeros((len(numeric), len(FREQUENCY_BANDS)), dtype=np.float32)
    for col_idx, band in enumerate(FREQUENCY_BANDS):
        out[:, col_idx] = (numeric == band).astype(np.float32)
    return out


def build_word_feature_matrix(words: pd.DataFrame, embeddings: np.ndarray, frequency: pd.DataFrame, feature_set: str = "legacy") -> np.ndarray:
    if words.empty:
        return np.zeros((0, embeddings.shape[1] + 2), dtype=np.float32)
    available_columns = ["word_id"] + [column for column in frequency.columns if column != "word_id"]
    freq = frequency[available_columns].copy() if "word_id" in frequency.columns else pd.DataFrame(columns=["word_id"])
    merged = words[["word_id", "length"]].merge(freq, on="word_id", how="left")
    feature_blocks: list[np.ndarray] = [embeddings.astype(np.float32)]
    if feature_set == "fasttext_only":
        return feature_blocks[0]
    length = _normalized_numeric_column(merged["length"])
    log_frequency = _raw_numeric_column(merged["log_frequency"]) if "log_frequency" in merged.columns else np.zeros((len(merged), 1), dtype=np.float32)
    if feature_set == "legacy":
        return np.concatenate([feature_blocks[0], merged["length"].fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1), log_frequency], axis=1)
    feature_blocks.append(length)
    if feature_set in {"freq", "rich"}:
        for column in FREQUENCY_FEATURE_COLUMNS:
            values = merged[column] if column in merged.columns else pd.Series(np.nan, index=merged.index)
            feature_blocks.append(_normalized_numeric_column(values))
        band_values = merged["frequency_band"] if "frequency_band" in merged.columns else pd.Series(np.nan, index=merged.index)
        feature_blocks.append(_frequency_band_matrix(band_values))
    else:
        feature_blocks.append(_normalized_numeric_column(merged["log_frequency"]) if "log_frequency" in merged.columns else np.zeros((len(merged), 1), dtype=np.float32))
    if feature_set in {"l2", "rich"}:
        for column in L2_FEATURE_COLUMNS:
            values = merged[column] if column in merged.columns else pd.Series(np.nan, index=merged.index)
            feature_blocks.append(_normalized_numeric_column(values))
    if feature_set not in {"legacy", "fasttext_only", "l2", "freq", "rich"}:
        raise ValueError(f"unknown feature_set={feature_set}")
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def build_response_frame(responses_static: pd.DataFrame, user_index: dict[str, int], word_index: dict[str, int]) -> pd.DataFrame:
    if responses_static.empty:
        return pd.DataFrame(columns=["user_idx", "word_idx", "label", "user_id", "word_id"])
    out = responses_static.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["word_id"] = out["word_id"].astype(str)
    out = out[out["word_id"].isin(word_index)]
    out["user_idx"] = out["user_id"].map(user_index)
    out["word_idx"] = out["word_id"].map(word_index)
    out["label"] = out["label"].astype(int)
    return out[["user_idx", "word_idx", "label", "user_id", "word_id"]]
