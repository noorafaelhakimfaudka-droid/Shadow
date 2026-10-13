# shadow_core/audit.py
"""
S2 — Audit Data (HANYA dijalankan pada data dev, setelah brankas dipisah) (PRD 6.2).
Aturan R11: Pemeriksaan validitas hidup di Core (kode), bukan di prompt agen.
Temuan bertingkat: blocker / peringatan / info.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import HistGradientBoostingClassifier


Severity = Literal["blocker", "peringatan", "info"]


@dataclass
class Finding:
    severity: Severity
    category: str
    column: Optional[str]
    message: str
    action_required: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    total_rows: int
    total_columns: int
    findings: List[Finding]
    blockers: List[Finding]
    warnings: List[Finding]
    info_list: List[Finding]
    column_stats: Dict[str, Dict[str, Any]]
    adversarial_auc: Optional[float] = None
    has_blocker: bool = False


def audit_dataset(
    df: pd.DataFrame,
    target_col: str,
    id_threshold: float = 0.90,
) -> AuditReport:
    """Melakukan audit kualitas data dan mendeteksi struktur tersembunyi pada data dev."""
    findings: List[Finding] = []
    col_stats: Dict[str, Dict[str, Any]] = {}
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0:
        findings.append(Finding(
            severity="blocker",
            category="empty_data",
            column=None,
            message="Dataset kosong (0 baris)! Eksekusi tidak dapat dilanjutkan.",
            action_required=True,
        ))
        return AuditReport(0, total_cols, findings, findings, [], [], {}, has_blocker=True)

    if target_col not in df.columns:
        findings.append(Finding(
            severity="blocker",
            category="missing_target",
            column=target_col,
            message=f"Kolom target '{target_col}' tidak ada dalam data dev!",
            action_required=True,
        ))

    # 1. Duplikasi Baris Persis
    exact_dups = int(df.duplicated().sum())
    if exact_dups > 0:
        dup_ratio = exact_dups / total_rows
        sev: Severity = "blocker" if dup_ratio > 0.05 else "peringatan"
        findings.append(Finding(
            severity=sev,
            category="duplicate_rows",
            column=None,
            message=f"Ditemukan {exact_dups} ({dup_ratio:.1%}) baris duplikat persis.",
            action_required=sev == "blocker",
            details={"count": exact_dups, "ratio": dup_ratio},
        ))

    feature_cols = [c for c in df.columns if c != target_col]

    # 2. Audit Tiap Kolom
    for col in feature_cols:
        s = df[col]
        n_missing = int(s.isna().sum())
        missing_ratio = n_missing / total_rows
        n_unique = int(s.nunique(dropna=True))
        unique_ratio = n_unique / total_rows if total_rows > 0 else 0.0
        dtype_str = str(s.dtype)

        col_stats[col] = {
            "dtype": dtype_str,
            "missing_count": n_missing,
            "missing_ratio": missing_ratio,
            "unique_count": n_unique,
            "unique_ratio": unique_ratio,
        }

        # 2a. Kolom Konstan (0 varians)
        if n_unique <= 1:
            findings.append(Finding(
                severity="blocker",
                category="constant_column",
                column=col,
                message=f"Kolom '{col}' hanya memiliki {n_unique} nilai unik (konstan). Harus dibuang.",
                action_required=True,
            ))

        # 2b. Missing Value Ekstrem (> 90%)
        elif missing_ratio > 0.90:
            findings.append(Finding(
                severity="peringatan",
                category="high_missing",
                column=col,
                message=f"Kolom '{col}' memiliki {missing_ratio:.1%} nilai hilang. Pertimbangkan pembuangan.",
                action_required=False,
                details={"missing_ratio": missing_ratio},
            ))

        # 2c. Kolom Mirip ID / Kardinalitas > 90%
        elif unique_ratio > id_threshold and total_rows > 50:
            findings.append(Finding(
                severity="peringatan",
                category="id_like_column",
                column=col,
                message=f"Kolom '{col}' memiliki rasio unik {unique_ratio:.1%}, terindikasi sebagai ID/kunci primer.",
                action_required=False,
                details={"unique_ratio": unique_ratio},
            ))

    # 3. Urutan Baris Berhubungan dengan Target (Struktur Tersembunyi)
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        clean_y = df[target_col].dropna()
        if len(clean_y) > 10:
            row_idx = np.arange(len(clean_y))
            corr = np.corrcoef(row_idx, clean_y)[0, 1]
            if not np.isnan(corr) and abs(corr) > 0.35:
                findings.append(Finding(
                    severity="peringatan",
                    category="row_order_leakage",
                    column=target_col,
                    message=f"Urutan baris berkorelasi dengan target (r = {corr:.2f}). Urutan data tidak boleh diacak sembarangan.",
                    details={"correlation": float(corr)},
                ))

    # Klasifikasi temuan
    blockers = [f for f in findings if f.severity == "blocker"]
    warnings = [f for f in findings if f.severity == "peringatan"]
    info_list = [f for f in findings if f.severity == "info"]

    return AuditReport(
        total_rows=total_rows,
        total_columns=total_cols,
        findings=findings,
        blockers=blockers,
        warnings=warnings,
        info_list=info_list,
        column_stats=col_stats,
        has_blocker=len(blockers) > 0,
    )
