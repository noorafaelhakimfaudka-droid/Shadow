# shadow_core/framing.py
"""
S1 — Pembingkaian Masalah dan Metrik (PRD 6.1).
Mendeteksi jenis tugas (klasifikasi biner, multikelas, regresi),
menetapkan metrik utama dan pendamping wajib sesuai Aturan 6.1,
menetapkan margin praktis δ, dan menyiapkan baseline wajib.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd


TaskType = Literal["binary_classification", "multiclass_classification", "regression"]


@dataclass
class FramingResult:
    target_column: str
    goal: str
    task_type: TaskType
    primary_metric: str
    companion_metrics: List[str]
    practical_margin_delta: float
    is_imbalanced: bool
    minority_class_ratio: Optional[float]
    class_distribution: Optional[Dict[str, int]]
    target_iqr: Optional[float] = None
    baseline_dummy: str = ""
    baseline_linear: str = ""
    warnings: List[str] = field(default_factory=list)


def detect_task_type(y: pd.Series) -> Tuple[TaskType, Dict[str, Any], List[str]]:
    """Mendeteksi jenis tugas machine learning dari kolom target."""
    warnings = []
    clean_y = y.dropna()
    n_unique = clean_y.nunique()
    total = len(clean_y)

    if total == 0:
        raise ValueError("Kolom target kosong atau seluruh nilainya NaN!")

    if n_unique <= 1:
        raise ValueError(f"Target hanya memiliki {n_unique} nilai unik; tidak dapat dimodelkan!")

    # Cek tipe data
    is_numeric = pd.api.types.is_numeric_dtype(clean_y)

    if n_unique == 2:
        task_type: TaskType = "binary_classification"
    elif not is_numeric or clean_y.dtype == "object" or clean_y.dtype.name == "category":
        task_type = "multiclass_classification"
    else:
        # Numerik: cek apakah diskrit dengan sedikit nilai unik
        if n_unique <= 10 and (n_unique / total) < 0.05:
            # Nilai unik sedikit, mungkin klasifikasi
            task_type = "multiclass_classification"
            warnings.append(
                f"Target bertipe numerik tetapi hanya memiliki {n_unique} nilai unik. Dideteksi sebagai multikelas."
            )
        else:
            task_type = "regression"

    # Detail distribusi
    info: Dict[str, Any] = {"n_unique": n_unique}
    if task_type in ("binary_classification", "multiclass_classification"):
        counts = clean_y.value_counts()
        proportions = clean_y.value_counts(normalize=True)
        info["class_distribution"] = counts.to_dict()
        info["minority_ratio"] = float(proportions.min())
        info["is_imbalanced"] = bool(info["minority_ratio"] < 0.20)
    else:
        q75, q25 = np.percentile(clean_y, [75, 25])
        info["iqr"] = float(q75 - q25)
        info["is_imbalanced"] = False
        info["minority_ratio"] = None

    return task_type, info, warnings


def frame_problem(
    data: pd.DataFrame,
    target_column: str,
    goal: str = "",
    cost_sensitive_probabilities: bool = False,
    override_primary_metric: Optional[str] = None,
    delta_override: Optional[float] = None,
) -> FramingResult:
    """Membingkai permasalahan ML, menentukan metrik, δ, dan baseline (S1)."""
    if target_column not in data.columns:
        raise KeyError(f"Kolom target '{target_column}' tidak ditemukan di dataset!")

    y = data[target_column]
    task_type, info, warnings = detect_task_type(y)

    is_imbalanced = info.get("is_imbalanced", False)
    minority_ratio = info.get("minority_ratio")
    class_dist = info.get("class_distribution")
    target_iqr = info.get("iqr")

    # Penetapan Aturan Metrik 6.1
    if task_type == "binary_classification":
        baseline_dummy = "DummyClassifier(strategy='prior')"
        baseline_linear = "LogisticRegression(max_iter=1000)"
        if cost_sensitive_probabilities:
            default_primary = "log_loss"
            default_companions = ["brier_score", "calibration_curve", "pr_auc"]
        elif is_imbalanced:
            default_primary = "pr_auc"
            default_companions = ["recall_at_precision", "f1", "balanced_accuracy"]
            warnings.append(
                f"Data tidak seimbang (minoritas {minority_ratio:.1%}). PR-AUC wajib sebagai metrik utama; akurasi dilarang sebagai metrik tunggal."
            )
        else:
            default_primary = "balanced_accuracy"
            default_companions = ["macro_f1", "log_loss", "confusion_matrix"]
        delta = 0.01

    elif task_type == "multiclass_classification":
        baseline_dummy = "DummyClassifier(strategy='prior')"
        baseline_linear = "LogisticRegression(max_iter=1000)"
        if is_imbalanced:
            default_primary = "macro_f1"
            default_companions = ["balanced_accuracy", "log_loss", "confusion_matrix"]
            warnings.append(f"Data multikelas tidak seimbang (minoritas {minority_ratio:.1%}). Macro-F1 sebagai metrik utama.")
        else:
            default_primary = "macro_f1"
            default_companions = ["balanced_accuracy", "log_loss", "confusion_matrix"]
        delta = 0.01

    else:  # regression
        baseline_dummy = "DummyRegressor(strategy='median')"
        baseline_linear = "Ridge()"
        default_primary = "mae"
        default_companions = ["rmse", "r2", "residual_distribution"]
        # delta = 1% dari rentang antarkuartil (IQR)
        iqr_val = target_iqr if target_iqr and target_iqr > 0 else 1.0
        delta = 0.01 * iqr_val

    primary_metric = override_primary_metric or default_primary
    if delta_override is not None:
        delta = delta_override

    # Peringatan bila operator memilih akurasi saja pada data tidak seimbang
    if primary_metric == "accuracy" and is_imbalanced:
        warnings.append("PERINGATAN: Memilih akurasi pada data tidak seimbang melanggar Aturan 6.1!")

    return FramingResult(
        target_column=target_column,
        goal=goal,
        task_type=task_type,
        primary_metric=primary_metric,
        companion_metrics=default_companions,
        practical_margin_delta=round(delta, 5),
        is_imbalanced=is_imbalanced,
        minority_class_ratio=minority_ratio,
        class_distribution=class_dist,
        target_iqr=target_iqr,
        baseline_dummy=baseline_dummy,
        baseline_linear=baseline_linear,
        warnings=warnings,
    )
