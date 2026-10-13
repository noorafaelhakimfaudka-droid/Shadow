# shadow_core/leakage/l8_index.py
"""
L8 — Detektor Kebocoran Indeks / ID / Urutan (PRD Bagian 7 & Lampiran C.2 #1).
Mendeteksi kolom yang berperilaku seperti identifier unik, indeks baris,
atau urutan auto-increment (kardinalitas mendekati 100% atau monotonik sempurna)
yang tidak layak digunakan sebagai fitur prediktif.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd


@dataclass
class L8Finding:
    column: str
    reason: str
    uniqueness_ratio: float
    is_monotonic: bool
    is_blocker: bool
    description: str


@dataclass
class L8Report:
    passed: bool
    findings: List[L8Finding]
    limitations: str = (
        "Detektor L8 mendeteksi identifier berdasar kardinalitas tinggi dan keterurutan monotonik; "
        "tidak dapat mendeteksi ID bertipe hashing terselubung atau entitas sah yang memang berdistribusi unik."
    )


ID_REGEX = re.compile(r"^(id|uuid|guid|index|row_id|record_id|no|nomor|cust_id|customer_id|user_id|patient_id|seq|num)$", re.IGNORECASE)
ID_SUFFIX_REGEX = re.compile(r"(_id|_uuid|_no|_num|_index)$", re.IGNORECASE)


def detect_l8_index_leakage(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    uniqueness_threshold: float = 0.98,
    min_rows: int = 50,
) -> L8Report:
    """
    Mendeteksi kolom ID / indeks baris / sequence.
    """
    findings: List[L8Finding] = []
    n = len(df)
    cols = [c for c in df.columns if c != target_col]

    if n < 10:
        return L8Report(passed=True, findings=[])

    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        n_unique = s.nunique()
        uniq_ratio = float(n_unique / len(s))
        
        is_monotonic = False
        if pd.api.types.is_numeric_dtype(s) and len(s) >= 10:
            vals = s.to_numpy()
            diffs = np.diff(vals)
            if np.all(diffs > 0) or np.all(diffs < 0):
                is_monotonic = True

        name_match = bool(ID_REGEX.match(col) or ID_SUFFIX_REGEX.search(col))

        is_leak = False
        reasons: List[str] = []

        is_float = pd.api.types.is_float_dtype(s)
        is_integer = pd.api.types.is_integer_dtype(s)
        is_string_or_obj = pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)

        if n >= min_rows and uniq_ratio >= uniqueness_threshold:
            if is_string_or_obj or name_match:
                is_leak = True
                reasons.append(f"Kardinalitas identifier sangat tinggi ({uniq_ratio:.1%}) mendekati 1 baris 1 nilai unik")
            elif is_integer and (is_monotonic or name_match or int(s.max() - s.min()) <= n + 10):
                is_leak = True
                reasons.append("Kolom integer berurutan/berperilaku seperti auto-increment index")

        if is_monotonic and (is_integer or name_match):
            is_leak = True
            reasons.append("Nilai numerik terurut monotonik sempurna mengikuti indeks baris")

        if is_leak:
            findings.append(
                L8Finding(
                    column=col,
                    reason="; ".join(reasons),
                    uniqueness_ratio=uniq_ratio,
                    is_monotonic=is_monotonic,
                    is_blocker=bool(uniq_ratio >= 0.99 or (is_monotonic and name_match)),
                    description=f"Kolom '{col}' terindikasi sebagai Identifier/Indeks baris ({'; '.join(reasons)})."
                )
            )

    passed = len(findings) == 0
    return L8Report(passed=passed, findings=findings)
