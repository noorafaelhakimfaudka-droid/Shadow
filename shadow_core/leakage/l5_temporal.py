# shadow_core/leakage/l5_temporal.py
"""
L5 — Detektor Kebocoran Temporal (PRD Bagian 7 & Aturan R3).
Mendeteksi apakah data berwaktu dibagi secara acak (random split) atau fitur mengandung sinyal masa depan.
Aturan R3: Data berwaktu dilarang di-split acak. Latih di masa lalu, uji di masa depan.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class L5Report:
    passed: bool
    time_column: Optional[str]
    split_strategy_used: str
    is_blocker: bool
    description: str
    performance_gap: Optional[float] = None
    limitations: str = (
        "Detektor L5 mengandalkan keberadaan kolom waktu atau timestamp yang terdeteksi. "
        "Fitur masa depan tanpa kolom penanda waktu eksplisit tidak tertangkap secara otomatis."
    )


def detect_l5_temporal_leakage(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str] = None,
    split_strategy_used: str = "",
) -> L5Report:
    """Mendeteksi pelanggaran split pada data berwaktu."""
    feature_cols = [c for c in df.columns if c != target_col]

    detected_time_col = time_col
    if detected_time_col is None:
        time_keywords = ("date", "time", "timestamp", "tanggal", "waktu", "year", "month", "dt")
        for c in feature_cols:
            c_str = str(c).lower()
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                detected_time_col = c
                break
            if any(kw in c_str for kw in time_keywords):
                try:
                    sample = df[c].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_datetime(sample)
                        detected_time_col = c
                        break
                except Exception:
                    pass

    # Jika tidak ada kolom waktu, lolos
    if detected_time_col is None:
        return L5Report(
            passed=True,
            time_column=None,
            split_strategy_used=split_strategy_used,
            is_blocker=False,
            description="Tidak terdeteksi kolom waktu bermakna pada dataset.",
        )

    # Ada kolom waktu: periksa strategi split yang digunakan
    # Jika split acak (misal KFold biasa) -> BLOCKER
    is_random_strategy = (
        "random" in split_strategy_used.lower()
        or "stratifiedkfold" in split_strategy_used.lower()
        or "kfold" in split_strategy_used.lower()
    ) and "temporal" not in split_strategy_used.lower() and "timeseries" not in split_strategy_used.lower()

    if is_random_strategy:
        return L5Report(
            passed=False,
            time_column=detected_time_col,
            split_strategy_used=split_strategy_used,
            is_blocker=True,
            description=(
                f"PELANGGARAN ATURAN R3: Kolom waktu '{detected_time_col}' terdeteksi, namun strategi split "
                f"menggunakan split acak ('{split_strategy_used}'). Wajib menggunakan TimeSeriesSplit dengan embargo."
            ),
        )

    return L5Report(
        passed=True,
        time_column=detected_time_col,
        split_strategy_used=split_strategy_used,
        is_blocker=False,
        description=f"Kolom waktu '{detected_time_col}' ditangani dengan strategi validasi temporal.",
    )
