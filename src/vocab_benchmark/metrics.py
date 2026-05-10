from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def entropy(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    nll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    brier = float(brier_score_loss(y_true, y_prob))
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else math.nan
    ap_known = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else math.nan
    ap_unknown = float(average_precision_score(1 - y_true, 1 - y_prob)) if len(np.unique(y_true)) > 1 else math.nan
    h = entropy(y_prob)
    uncertain = float(np.mean((y_prob > 0.4) & (y_prob < 0.6)))
    confident = float(np.mean((y_prob <= 0.2) | (y_prob >= 0.8)))
    return {
        "nll": nll,
        "brier": brier,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "average_precision_known": ap_known,
        "average_precision_unknown": ap_unknown,
        "expected_calibration_error": expected_calibration_error(y_true, y_prob, n_bins=10),
        "mean_predictive_entropy": float(np.mean(h)),
        "uncertain_fraction_0_4_0_6": uncertain,
        "confident_fraction_0_2_0_8": confident,
    }
