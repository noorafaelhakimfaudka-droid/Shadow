# shadow_core/leakage/canary.py
"""
Kanari Permutasi (PRD 7.1 & Aturan [K]).
Menjalankan seluruh alur pemodelan dengan label diacak (≥ 5 permutasi).
Pada label acak, model tidak boleh bisa belajar. Skor di atas batas tebakan acak mengindikasikan kebocoran prosedural.
Ambang alarm: skor > chance + max(0.05, 3 * SE).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, r2_score


@dataclass
class CanaryReport:
    passed: bool
    canary_scores: List[float]
    mean_score: float
    se_score: float
    chance_level: float
    alarm_threshold: float
    is_blocker: bool
    description: str
    limitations: str = (
        "Kanari permutasi mendeteksi kebocoran prosedural (misal target encoding / imputasi di luar fold). "
        "Kanari tidak menangkap kebocoran di mana fitur mentah sudah membawa label (L1) atau duplikat (L2)."
    )


def run_permutation_canary(
    pipeline_estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    is_classification: bool = True,
    n_permutations: int = 5,
    cv_folds: int = 5,
    seed: int = 42,
) -> CanaryReport:
    """Menjalankan pengujian kanari permutasi label acak."""
    rng = np.random.RandomState(seed)
    y_series = pd.Series(y)
    if is_classification:
        # Seragamkan target klasifikasi ke string agar aman dari perbandingan tipe campuran
        y_arr = y_series.astype(str).to_numpy()
    else:
        y_arr = pd.to_numeric(y_series, errors="coerce").fillna(0.0).to_numpy()

    n_samples = len(y_arr)
    chance_level = 0.50 if is_classification else 0.00
    scores = []

    for p_idx in range(n_permutations):
        # Acak label target
        y_shuffled = rng.permutation(y_arr)

        n_unique_y = pd.Series(y_shuffled).nunique()
        if is_classification and 2 <= n_unique_y <= 10:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + p_idx)
            splits = cv.split(X, y_shuffled)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed + p_idx)
            splits = cv.split(X)

        fold_scores = []
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_shuffled[train_idx], y_shuffled[val_idx]

            model = clone(pipeline_estimator)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            if is_classification:
                sc = balanced_accuracy_score(y_val, preds)
            else:
                sc = r2_score(y_val, preds)
            fold_scores.append(sc)

        scores.append(float(np.mean(fold_scores)))

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    se_score = std_score / np.sqrt(len(scores))

    # Formula PRD 7.1: chance + max(0.05, 3 * SE)
    alarm_threshold = chance_level + max(0.05, 3.0 * se_score)
    passed = bool(mean_score <= alarm_threshold)
    is_blocker = not passed

    if passed:
        desc = (
            f"Kanari permutasi lolos. Skor label acak: {mean_score:.4f} ± {se_score:.4f} "
            f"(di bawah ambang batas alarm {alarm_threshold:.4f}, baseline acak {chance_level:.2f})."
        )
    else:
        desc = (
            f"ALARM KEBOCORAN PROSEDURAL: Model mampu memprediksi label acak dengan skor {mean_score:.4f} "
            f"(melebihi ambang alarm {alarm_threshold:.4f}). Ada kontaminasi label dalam alur pra-pemrosesan!"
        )

    return CanaryReport(
        passed=passed,
        canary_scores=[round(s, 4) for s in scores],
        mean_score=round(mean_score, 4),
        se_score=round(se_score, 4),
        chance_level=chance_level,
        alarm_threshold=round(alarm_threshold, 4),
        is_blocker=is_blocker,
        description=desc,
    )
