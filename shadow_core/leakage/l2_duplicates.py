# shadow_core/leakage/l2_duplicates.py
"""
L2 — Detektor Baris Duplikat Lintas Split (PRD Bagian 7).
Mendeteksi baris pada data uji yang identik atau hampir identik dengan data latih.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class L2Report:
    passed: bool
    exact_duplicate_count: int
    exact_duplicate_ratio: float
    fuzzy_duplicate_count: int
    fuzzy_duplicate_ratio: float
    is_blocker: bool
    description: str
    limitations: str = (
        "Detektor L2 tidak menangkap duplikasi semantik (misal nama jalan dengan singkatan berbeda "
        "atau representasi teks berbeda dengan arti sama)."
    )


def compute_row_hashes(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Menghitung hash SHA-256 untuk setiap baris fitur."""
    serialized = df[cols].astype(str).agg("|".join, axis=1)
    return serialized.apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


def detect_l2_cross_split_duplicates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    blocker_threshold: float = 0.01,
) -> L2Report:
    """Mendeteksi kontaminasi baris duplikat antara train fold dan test fold."""
    if feature_cols is None:
        common = set(train_df.columns).intersection(test_df.columns)
        feature_cols = sorted(list(common))

    n_test = len(test_df)
    if n_test == 0 or len(train_df) == 0:
        return L2Report(True, 0, 0.0, 0, 0.0, False, "Data kosong, tidak ada duplikat.")

    # 1. Duplikat Persis
    train_hashes = set(compute_row_hashes(train_df, feature_cols))
    test_hashes = compute_row_hashes(test_df, feature_cols)

    exact_matches = int(test_hashes.isin(train_hashes).sum())
    exact_ratio = exact_matches / n_test

    # 2. Duplikat Fuzzy (Numerik dibulatkan)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    fuzzy_matches = 0
    if len(numeric_cols) > 0:
        rounded_train = train_df[numeric_cols].round(2)
        rounded_test = test_df[numeric_cols].round(2)
        train_fuz = set(rounded_train.astype(str).agg("|".join, axis=1))
        test_fuz = rounded_test.astype(str).agg("|".join, axis=1)
        fuzzy_matches = int(test_fuz.isin(train_fuz).sum())

    fuzzy_ratio = fuzzy_matches / n_test if n_test > 0 else 0.0

    is_blocker = exact_ratio > blocker_threshold or (exact_matches > 0 and n_test < 100)
    passed = exact_matches == 0

    desc = (
        f"Ditemukan {exact_matches} ({exact_ratio:.2%}) baris duplikat persis dan "
        f"{fuzzy_matches} ({fuzzy_ratio:.2%}) duplikat fuzzy lintas split latih-uji."
    )

    return L2Report(
        passed=passed,
        exact_duplicate_count=exact_matches,
        exact_duplicate_ratio=round(exact_ratio, 4),
        fuzzy_duplicate_count=fuzzy_matches,
        fuzzy_duplicate_ratio=round(fuzzy_ratio, 4),
        is_blocker=is_blocker,
        description=desc,
    )
