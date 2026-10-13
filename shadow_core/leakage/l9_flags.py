# shadow_core/leakage/l9_flags.py
"""
L9 — Detektor Konfirmasi Flag Pasca-Kejadian / Flag Kebocoran (PRD Bagian 7).
Mendeteksi kolom yang merupakan indikator status pasca-kejadian (post-outcome flags)
seperti 'cancellation_reason', 'refund_amount', 'churn_date', 'discharge_summary'
yang terisi hanya jika target bernilai positif atau terjadi setelah peristiwa target.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd


@dataclass
class L9Finding:
    column: str
    flag_type: str
    condition_detail: str
    is_blocker: bool
    description: str


@dataclass
class L9Report:
    passed: bool
    findings: List[L9Finding]
    limitations: str = (
        "Detektor L9 berbasis heuristik nama dan pola keterisian kondisional terhadap target; "
        "tidak menggantikan pemahaman domain bisnis terhadap alur waktu pencatatan data."
    )


POST_OUTCOME_REGEX = re.compile(
    r"(reason|alasan|date_exit|tanggal_keluar|refund|cancel|batal|discharge|survival|death|klaim_status|tindak_lanjut)",
    re.IGNORECASE
)


def detect_l9_flag_leakage(
    df: pd.DataFrame,
    target_col: str,
) -> L9Report:
    """
    Mendeteksi flag pasca-kejadian atau kolom yang hanya terisi saat target terjadi.
    """
    if target_col not in df.columns:
        return L9Report(passed=True, findings=[])

    findings: List[L9Finding] = []
    y = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]

    # Cek pola keterisian bersyarat (Conditional Non-Null Sparsity)
    # Jika kolom X terisi (non-null) hampir 100% pada y == 1 dan null pada y == 0
    if y.nunique() == 2:
        classes = list(y.unique())
        pos_class = classes[1] if (y == classes[1]).sum() <= (y == classes[0]).sum() else classes[0]
        neg_class = classes[0] if pos_class == classes[1] else classes[1]

        y_pos = (y == pos_class)
        y_neg = (y == neg_class)

        for col in feature_cols:
            s = df[col]
            if s.isna().sum() > 0:
                pos_fill_rate = float(s[y_pos].notna().mean())
                neg_fill_rate = float(s[y_neg].notna().mean())

                # Sangat terisi saat pos tapi hampir kosong saat neg (e.g. churn_reason, refund_date)
                if pos_fill_rate >= 0.80 and neg_fill_rate <= 0.05:
                    findings.append(
                        L9Finding(
                            column=col,
                            flag_type="conditional_presence",
                            condition_detail=f"Terisi {pos_fill_rate:.1%} pada kelas target, tapi hanya {neg_fill_rate:.1%} pada kelas kontrol",
                            is_blocker=True,
                            description=f"Kolom '{col}' terindikasi sebagai flag pasca-kejadian (hampir hanya terisi saat target positif terjadi)."
                        )
                    )
                    continue

    # Heuristik pola nama pasca kejadian
    for col in feature_cols:
        if POST_OUTCOME_REGEX.search(col) and not any(f.column == col for f in findings):
            findings.append(
                L9Finding(
                    column=col,
                    flag_type="post_outcome_name",
                    condition_detail=f"Nama kolom cocok dengan pola variabel peristiwa pasca-kejadian '{POST_OUTCOME_REGEX.pattern}'",
                    is_blocker=False,
                    description=f"Kolom '{col}' dicurigai sebagai informasi pasca-kejadian berdasarkan penamaan."
                )
            )

    passed = len(findings) == 0
    return L9Report(passed=passed, findings=findings)
