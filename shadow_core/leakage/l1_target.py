# shadow_core/leakage/l1_target.py
"""
L1 — Detektor Fitur Turunan Target (PRD Bagian 7).
Mendeteksi apakah ada fitur tunggal yang terlalu prediktif terhadap target
(AUC > 0.99 atau R^2 > 0.99, korelasi Pearson/Spearman > 0.98).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score


@dataclass
class L1Finding:
    column: str
    metric_name: str
    metric_value: float
    is_blocker: bool
    description: str


@dataclass
class L1Report:
    passed: bool
    findings: List[L1Finding]
    threshold_auc: float
    threshold_r2: float
    limitations: str = (
        "Detektor L1 tidak mendeteksi kebocoran kombinasi multi-fitur yang secara individual "
        "memiliki korelasi rendah, serta dapat menghasilkan alarm pada fitur sah yang sangat kuat."
    )


def detect_l1_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    threshold_auc: float = 0.99,
    threshold_r2: float = 0.99,
    threshold_corr: float = 0.98,
) -> L1Report:
    """Mendeteksi fitur turunan target tunggal."""
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' tidak ada dalam data!")

    y = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]
    findings: List[L1Finding] = []

    is_binary = bool(y.nunique() == 2)
    is_numeric_y = bool(pd.api.types.is_numeric_dtype(y))

    for col in feature_cols:
        s = df[col]
        # Skip jika konstan atau semuanya NaN
        if s.nunique(dropna=True) <= 1 or s.isna().all():
            continue

        # 1. Cek Korelasi bila keduanya numerik
        if is_numeric_y and pd.api.types.is_numeric_dtype(s):
            clean = df[[col, target_col]].dropna()
            if len(clean) > 5:
                # Cek standar deviasi
                if clean[col].std() > 0 and clean[target_col].std() > 0:
                    corr = abs(clean[col].corr(clean[target_col]))
                    if corr >= threshold_corr:
                        findings.append(L1Finding(
                            column=col,
                            metric_name="abs_correlation",
                            metric_value=round(corr, 4),
                            is_blocker=True,
                            description=f"Fitur '{col}' memiliki korelasi linear ekstrem ({corr:.4f}) dengan target.",
                        ))
                        continue

        # 2. Cek AUC tunggal (untuk klasifikasi biner)
        if is_binary and pd.api.types.is_numeric_dtype(s):
            clean = df[[col, target_col]].dropna()
            if len(clean) > 10 and clean[target_col].nunique() == 2:
                try:
                    auc = roc_auc_score(clean[target_col], clean[col])
                    # Handle inverted score
                    effective_auc = max(auc, 1.0 - auc)
                    if effective_auc >= threshold_auc:
                        findings.append(L1Finding(
                            column=col,
                            metric_name="roc_auc",
                            metric_value=round(effective_auc, 4),
                            is_blocker=True,
                            description=f"Fitur tunggal '{col}' memiliki AUC ekstrem ({effective_auc:.4f}) terhadap target.",
                        ))
                        continue
                except Exception:
                    pass

        # 3. Cek R2 tunggal (untuk regresi)
        if not is_binary and is_numeric_y and pd.api.types.is_numeric_dtype(s):
            clean = df[[col, target_col]].dropna()
            if len(clean) > 10 and clean[target_col].std() > 0:
                try:
                    # Model linear sederhana y ~ a*x + b
                    p = np.polyfit(clean[col], clean[target_col], deg=1)
                    y_pred = np.polyval(p, clean[col])
                    r2 = r2_score(clean[target_col], y_pred)
                    if r2 >= threshold_r2:
                        findings.append(L1Finding(
                            column=col,
                            metric_name="r2_score",
                            metric_value=round(r2, 4),
                            is_blocker=True,
                            description=f"Fitur tunggal '{col}' memprediksi target dengan R² ekstrem ({r2:.4f}).",
                        ))
                except Exception:
                    pass

    passed = len(findings) == 0
    return L1Report(
        passed=passed,
        findings=findings,
        threshold_auc=threshold_auc,
        threshold_r2=threshold_r2,
    )
