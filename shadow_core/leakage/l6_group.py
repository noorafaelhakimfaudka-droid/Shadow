# shadow_core/leakage/l6_group.py
"""
L6 — Detektor Kebocoran Grup (PRD Bagian 7).
Mendeteksi apakah entitas grup yang sama (misal user_id, patient_id)
hadir di dalam data latih dan data uji secara bersamaan.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


@dataclass
class L6Report:
    passed: bool
    group_column: Optional[str]
    overlapping_groups_count: int
    overlapping_groups_sample: List[str]
    is_blocker: bool
    description: str
    limitations: str = (
        "Detektor L6 memerlukan entitas grup terkodekan dalam kolom fitur. "
        "Relasi kelompok laten tanpa ID eksplisit tidak dapat dideteksi secara otomatis."
    )


def detect_l6_group_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: Optional[str] = None,
) -> L6Report:
    """Mendeteksi apakah ada ID grup yang bocor antara train dan test split."""
    candidate_cols = [group_col] if group_col else []

    if not candidate_cols:
        # Cari kolom mirip grup
        group_keywords = ("id", "user", "customer", "patient", "client", "group", "account")
        for col in train_df.columns:
            col_l = str(col).lower()
            if any(kw in col_l for kw in group_keywords):
                n_uniq = train_df[col].nunique()
                if 1 < n_uniq < len(train_df):
                    candidate_cols.append(col)

    if not candidate_cols:
        return L6Report(
            passed=True,
            group_column=None,
            overlapping_groups_count=0,
            overlapping_groups_sample=[],
            is_blocker=False,
            description="Tidak terdeteksi kolom grup/ID pada dataset.",
        )

    # Periksa tumpang tindih untuk setiap kandidat grup
    worst_col = None
    max_overlap = 0
    overlap_sample: List[str] = []

    for col in candidate_cols:
        if col in test_df.columns:
            train_groups: Set[Any] = set(train_df[col].dropna().unique())
            test_groups: Set[Any] = set(test_df[col].dropna().unique())
            intersection = train_groups.intersection(test_groups)

            if len(intersection) > max_overlap:
                max_overlap = len(intersection)
                worst_col = col
                overlap_sample = [str(x) for x in list(intersection)[:5]]

    if max_overlap > 0:
        return L6Report(
            passed=False,
            group_column=worst_col,
            overlapping_groups_count=max_overlap,
            overlapping_groups_sample=overlap_sample,
            is_blocker=True,
            description=(
                f"PELANGGARAN STRATEGI GRUP: Ditemukan {max_overlap} entitas dari kolom grup '{worst_col}' "
                f"yang hadir di kedua split latih dan uji (misal: {overlap_sample}). Wajib menggunakan GroupKFold!"
            ),
        )

    return L6Report(
        passed=True,
        group_column=candidate_cols[0],
        overlapping_groups_count=0,
        overlapping_groups_sample=[],
        is_blocker=False,
        description=f"Kolom grup '{candidate_cols[0]}' terisolasi sempurna antara split latih dan uji (0 tumpang tindih).",
    )
