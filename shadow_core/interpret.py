# shadow_core/interpret.py
"""
S9 — Interpretasi, Ketahanan, dan Learning Curve (PRD Bagian 6.8 & 6.9).
Menghitung Permutation Importance, variasi seed, uji stres,
learning curve, dan blok keterbatasan wajib.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, r2_score


@dataclass
class RobustnessReport:
    seed_scores: List[float]
    mean_score: float
    std_score: float
    max_delta: float
    noise_stressed_score: float
    is_fragile: bool
    description: str


@dataclass
class LearningCurveReport:
    fractions: List[float]
    train_scores: List[float]
    val_scores: List[float]
    plateau_reached: bool


def compute_permutation_importance_safe(
    model: Any,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    feature_names: List[str],
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Menghitung permutation importance pada data held-out.
    Wajib berlabel 'penjelasan model, bukan sebab-akibat' (Aturan R14).
    """
    perm = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    importances = []
    for i, name in enumerate(feature_names):
        importances.append({
            "feature": name,
            "mean_importance": round(float(perm.importances_mean[i]), 5),
            "std_importance": round(float(perm.importances_std[i]), 5),
        })

    importances.sort(key=lambda x: x["mean_importance"], reverse=True)
    return {
        "importances": importances,
        "label": "penjelasan model, bukan sebab-akibat (Aturan R14)",
        "scoring": scoring,
    }


def compute_learning_curves(
    pipeline_factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fractions: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    metric_fn: Any = None,
) -> LearningCurveReport:
    """Menghitung performa model pada fraksi data latih yang bertambah."""
    if metric_fn is None:
        metric_fn = lambda yt, yp: float(roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else r2_score(yt, yp))

    train_scores = []
    val_scores = []
    n_total = len(X_train)

    for frac in fractions:
        n_sub = max(10, int(n_total * frac))
        sub_X = X_train.iloc[:n_sub]
        sub_y = y_train.iloc[:n_sub]

        pipe = pipeline_factory()
        pipe.fit(sub_X, sub_y)

        # Train prediction
        try:
            preds_tr = pipe.predict_proba(sub_X)[:, 1]
        except Exception:
            preds_tr = pipe.predict(sub_X)

        # Val prediction
        try:
            preds_val = pipe.predict_proba(X_val)[:, 1]
        except Exception:
            preds_val = pipe.predict(X_val)

        train_scores.append(round(metric_fn(sub_y, preds_tr), 4))
        val_scores.append(round(metric_fn(y_val, preds_val), 4))

    # Cek plateau: apakah kenaikan dari 0.8 ke 1.0 < 0.01
    plateau = bool(abs(val_scores[-1] - val_scores[-2]) < 0.01)

    return LearningCurveReport(
        fractions=fractions,
        train_scores=train_scores,
        val_scores=val_scores,
        plateau_reached=plateau,
    )


def evaluate_robustness(
    pipeline_builder: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seeds: List[int] = [42, 101, 2024, 7, 99],
) -> RobustnessReport:
    """Evaluasi variasi performa model antar-seed dan di bawah derau."""
    scores = []
    for seed in seeds:
        pipe = pipeline_builder(seed=seed)
        pipe.fit(X_train, y_train)
        try:
            preds = pipe.predict_proba(X_val)[:, 1]
            score = float(roc_auc_score(y_val, preds))
        except Exception:
            preds = pipe.predict(X_val)
            score = float(r2_score(y_val, preds))
        scores.append(score)

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    max_d = float(np.max(scores) - np.min(scores))

    # Uji stres derau pada fitur numerik
    pipe_base = pipeline_builder(seed=42)
    pipe_base.fit(X_train, y_train)
    X_val_noise = X_val.copy()
    num_cols = X_val.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        noise = np.random.normal(0, 0.05 * (X_val[c].std() or 1.0), size=len(X_val))
        X_val_noise[c] = X_val[c] + noise

    try:
        preds_noise = pipe_base.predict_proba(X_val_noise)[:, 1]
        noise_score = float(roc_auc_score(y_val, preds_noise))
    except Exception:
        preds_noise = pipe_base.predict(X_val_noise)
        noise_score = float(r2_score(y_val, preds_noise))

    is_fragile = bool(max_d > 0.05 or (mean_s - noise_score) > 0.10)
    desc = (
        f"Variasi 5-seed: {mean_s:.4f} ± {std_s:.4f} (max delta: {max_d:.4f}). "
        f"Skor setelah 5% derau Gaussian: {noise_score:.4f}. "
    )
    if is_fragile:
        desc += "Peringatan: Model sensitif terhadap inisialisasi acak atau derau fitur."
    else:
        desc += "Model stabil terhadap variasi acak."

    return RobustnessReport(
        seed_scores=[round(s, 4) for s in scores],
        mean_score=round(mean_s, 4),
        std_score=round(std_s, 4),
        max_delta=round(max_d, 4),
        noise_stressed_score=round(noise_score, 4),
        is_fragile=is_fragile,
        description=desc,
    )
