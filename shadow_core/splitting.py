# shadow_core/splitting.py
"""
S3a — Pemisahan Brankas (dilakukan SEBELUM audit S2) [K].
Aturan R2: Uji tersegel dipisah pertama kali dan hanya disentuh satu kali.
Aturan R3: Data berwaktu dilarang di-split acak.
S3a.1: Deteksi struktur TANPA membaca target (target-blind).
S3a.2: Pemisahan fisik: dev.parquet dan sealed.bin terenkripsi.
"""

from __future__ import annotations
import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.model_selection import StratifiedKFold, KFold


@dataclass
class DevData:
    dev_df: pd.DataFrame
    split_strategy: str
    sealed_indices_sha256: str
    is_small_data: bool
    time_col: Optional[str]
    group_col: Optional[str]
    dev_path: Path
    sealed_path: Path
    n_dev: int
    n_sealed: int


def detect_structure_without_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[Optional[str], Optional[str], str]:
    """Mendeteksi kolom waktu dan grup HANYA dari fitur tanpa menyentuh isi target.
    Menghasilkan: (time_col, group_col, strategy_name)"""
    candidate_features = [c for c in df.columns if c != target_col]

    time_col: Optional[str] = None
    group_col: Optional[str] = None

    time_keywords = ("date", "time", "timestamp", "tanggal", "waktu", "year", "month", "dt")
    group_keywords = ("id", "user", "customer", "patient", "client", "group", "entity", "account")

    # 1. Deteksi waktu
    for col in candidate_features:
        s = df[col]
        col_lower = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(s):
            time_col = col
            break
        if any(kw in col_lower for kw in time_keywords):
            # Cek apakah bisa diparse tanggal
            try:
                sample = s.dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    time_col = col
                    break
            except Exception:
                pass

    # 2. Deteksi grup (ID berulang lintas baris)
    for col in candidate_features:
        if col == time_col:
            continue
        col_lower = str(col).lower()
        s = df[col]
        n_unique = s.nunique(dropna=True)
        total = len(s)
        if total > 0 and 1 < n_unique < total:
            ratio = n_unique / total
            # Kolom dengan ID berulang (setiap ID muncul rata-rata >= 1.5x)
            if any(kw in col_lower for kw in group_keywords) and ratio <= 0.8:
                group_col = col
                break
            elif ratio <= 0.5 and (s.dtype == "object" or pd.api.types.is_integer_dtype(s)):
                # Kolom kategorikal dengan perulangan sangat tinggi
                if any(kw in col_lower for kw in group_keywords):
                    group_col = col
                    break

    # Tentukan nama strategi
    if time_col is not None:
        strategy_name = f"TemporalSplit(time_col={time_col})"
    elif group_col is not None:
        strategy_name = f"GroupSplit(group_col={group_col})"
    else:
        strategy_name = "StratifiedSplit"

    return time_col, group_col, strategy_name


def pisah_brankas(
    csv_path: Path | str,
    target_col: str,
    run_dir: Path | str,
    seed: int = 42,
    sealed_fraction: float = 0.20,
) -> DevData:
    """Memisahkan data menjadi dev.parquet dan sealed.bin (terenkripsi).
    Proses pemanggil TIDAK memegang data tersegel setelah fungsi ini selesai."""
    csv_p = Path(csv_path)
    run_d = Path(run_dir)
    run_d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_p)
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' tidak ada di file {csv_p}!")

    total_rows = len(df)
    time_col, group_col, strategy_name = detect_structure_without_target(df, target_col)

    # Deteksi aturan data kecil (PRD 6.3 S3a.5)
    # Jika baris < 500 atau kelas minoritas di sealed < 50, gunakan nested CV
    is_small_data = False
    y = df[target_col]
    if total_rows < 500:
        is_small_data = True
    elif y.nunique() <= 10:
        min_class_count = y.value_counts().min()
        if (min_class_count * sealed_fraction) < 50:
            is_small_data = True

    if is_small_data:
        # Tidak ada split brankas terpisah, seluruh data menjadi dev
        dev_indices = np.arange(total_rows)
        sealed_indices = np.array([], dtype=int)
        strategy_name = f"NestedCV({strategy_name})"
    elif time_col is not None:
        # Temporal: urutkan berdasarkan waktu, bagian paling akhir menjadi sealed
        sorted_idx = df[time_col].sort_values(kind="stable").index.to_numpy()
        split_point = int(total_rows * (1.0 - sealed_fraction))
        dev_indices = sorted_idx[:split_point]
        sealed_indices = sorted_idx[split_point:]
    elif group_col is not None:
        # Group: kelompokkan entitas agar tidak pernah terbelah
        groups = df[group_col].unique()
        rng = np.random.RandomState(seed)
        shuffled_groups = rng.permutation(groups)
        cum_rows = 0
        sealed_groups = set()
        target_sealed_count = int(total_rows * sealed_fraction)
        for g in shuffled_groups:
            g_count = (df[group_col] == g).sum()
            sealed_groups.add(g)
            cum_rows += g_count
            if cum_rows >= target_sealed_count:
                break
        sealed_mask = df[group_col].isin(sealed_groups)
        sealed_indices = df[sealed_mask].index.to_numpy()
        dev_indices = df[~sealed_mask].index.to_numpy()
    else:
        # Stratified bila diskrit, else KFold
        rng = np.random.RandomState(seed)
        if y.nunique() <= 50 and pd.api.types.is_numeric_dtype(y) is False or y.nunique() <= 10:
            # Stratified
            classes = y.unique()
            sealed_list = []
            dev_list = []
            for c in classes:
                c_idx = df[y == c].index.to_numpy().copy()
                rng.shuffle(c_idx)
                n_seal = int(len(c_idx) * sealed_fraction)
                sealed_list.extend(c_idx[:n_seal])
                dev_list.extend(c_idx[n_seal:])
            dev_indices = np.array(dev_list)
            sealed_indices = np.array(sealed_list)
        else:
            shuffled = rng.permutation(total_rows)
            n_seal = int(total_rows * sealed_fraction)
            sealed_indices = shuffled[:n_seal]
            dev_indices = shuffled[n_seal:]

    # Hitung hash indeks tersegel (PRD S3a.2)
    sorted_sealed = sorted(sealed_indices.tolist())
    sealed_hash = hashlib.sha256(json.dumps(sorted_sealed).encode("utf-8")).hexdigest()

    # Pisahkan DataFrame
    dev_df = df.iloc[dev_indices].copy().reset_index(drop=True)
    dev_path = run_d / "dev.parquet"
    dev_df.to_parquet(dev_path, index=False)

    sealed_path = run_d / "sealed.bin"
    key_path = run_d / "vault.key"

    if len(sealed_indices) > 0:
        sealed_df = df.iloc[sealed_indices].copy().reset_index(drop=True)
        # Enkripsi sealed_df dengan Fernet AES-256
        key = Fernet.generate_key()
        with open(key_path, "wb") as kf:
            kf.write(key)

        fernet = Fernet(key)
        # Simpan payload parquet tersegel
        parquet_bytes = sealed_df.to_parquet(index=False)
        encrypted_bytes = fernet.encrypt(parquet_bytes)

        with open(sealed_path, "wb") as sf:
            sf.write(encrypted_bytes)

        # Hancurkan referensi sealed_df dari memori proses utama
        del sealed_df
        del parquet_bytes
        del encrypted_bytes
        gc.collect()
    else:
        # Data kecil / tidak ada data tersegel
        sealed_path.touch()

    # Hancurkan df utama dari memori
    del df
    gc.collect()

    return DevData(
        dev_df=dev_df,
        split_strategy=strategy_name,
        sealed_indices_sha256=sealed_hash,
        is_small_data=is_small_data,
        time_col=time_col,
        group_col=group_col,
        dev_path=dev_path,
        sealed_path=sealed_path,
        n_dev=len(dev_indices),
        n_sealed=len(sealed_indices),
    )
