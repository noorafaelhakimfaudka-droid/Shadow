# shadow_core/evaluation.py
"""
S8 — Evaluasi Akhir dan Diagnostik Metrik (PRD Bagian 6.8 & AC-10).
Menghitung kalibrasi (ECE, Brier), analisis galat per segmen,
dan selisih generalisasi (CV vs Uji Tersegel).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.calibration import calibration_curve


@dataclass
class CalibrationResult:
    brier_score: float
    ece: float  # Expected Calibration Error
    prob_true: List[float]
    prob_pred: List[float]


@dataclass
class GeneralizationGapResult:
    cv_metric: float
    test_metric: float
    gap: float
    threshold: float
    is_warning: bool
    description: str


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Menghitung kurva kalibrasi, Brier score, dan ECE."""
    brier = float(brier_score_loss(y_true, y_prob))

    # Binning untuk ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    prob_true_list = []
    prob_pred_list = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            p_true = float(np.mean(y_true[mask]))
            p_pred = float(np.mean(y_prob[mask]))
            prob_true_list.append(p_true)
            prob_pred_list.append(p_pred)
            ece += (bin_count / len(y_true)) * abs(p_true - p_pred)

    return CalibrationResult(
        brier_score=round(brier, 4),
        ece=round(float(ece), 4),
        prob_true=[round(v, 4) for v in prob_true_list],
        prob_pred=[round(v, 4) for v in prob_pred_list],
    )


def compute_generalization_gap(
    cv_metric: float,
    test_metric: float,
    threshold: float = 0.05,
    metric_name: str = "metric",
) -> GeneralizationGapResult:
    """
    Menghitung selisih generalisasi: cv_metric - test_metric.
    Sesuai PRD 6.8: bila CV - Uji Tersegel > 0.05 (skala 0-1) -> peringatan Mayor.
    """
    gap = cv_metric - test_metric
    is_warning = bool(gap > threshold)
    desc = (
        f"Selisih generalisasi ({metric_name}) = {gap:+.4f} (CV: {cv_metric:.4f}, Test: {test_metric:.4f}). "
    )
    if is_warning:
        desc += f"Peringatan Mayor: Terjadi penurunan generalisasi melebihi ambang batas {threshold}."
    else:
        desc += "Generalisasi terjaga dengan baik (dalam batas toleransi)."

    return GeneralizationGapResult(
        cv_metric=cv_metric,
        test_metric=test_metric,
        gap=round(gap, 4),
        threshold=threshold,
        is_warning=is_warning,
        description=desc,
    )


def compute_error_segments(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categorical_cols: List[str],
    top_k_errors: int = 5,
) -> Dict[str, Any]:
    """Menganalisis galat model pada subset data/segmen."""
    errors = np.abs(y_true - y_pred)
    res: Dict[str, Any] = {"segment_metrics": {}, "worst_indices": []}

    # Segmen per kolom kategorikal
    for col in categorical_cols:
        if col in df.columns:
            seg_perf = {}
            for val, group_idx in df.groupby(col).groups.items():
                if len(group_idx) >= 5:
                    seg_perf[str(val)] = {
                        "count": len(group_idx),
                        "mean_abs_error": round(float(np.mean(errors[group_idx])), 4),
                    }
            if seg_perf:
                res["segment_metrics"][col] = seg_perf

    worst_idx = np.argsort(errors)[::-1][:top_k_errors]
    res["worst_indices"] = worst_idx.tolist()
    return res
