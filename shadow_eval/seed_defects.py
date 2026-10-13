# shadow_eval/seed_defects.py
"""
Pembangkit Defect Tersemai & Kasus Uji Emas (PRD 10.2).
Menyediakan generator dataset sintetis dan manipulasi terarah untuk menguji V1 (L1, L2, L5, L6, L8),
kasus bersih (V2), serta kasus ekstrem (V10).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_clean_dataset(
    n_samples: int = 600,
    n_features: int = 10,
    task: str = "binary",
    seed: int = 42,
) -> pd.DataFrame:
    """Membangkitkan dataset kontrol bersih tanpa cacat kebocoran."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)

    if task == "binary":
        # Kombinasi linear moderat
        logits = 0.5 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.5, size=n_samples)
        probs = 1.0 / (1.0 + np.exp(-logits))
        df["target"] = (probs > 0.5).astype(int)
    elif task == "multiclass":
        scores = X[:, 0] + X[:, 1]
        df["target"] = pd.qcut(scores, q=3, labels=["kelas_A", "kelas_B", "kelas_C"]).astype(str)
    else:  # regression
        df["target"] = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.5, size=n_samples)

    return df


def seed_l1_target_copy(
    df: pd.DataFrame,
    target_col: str = "target",
    col_name: str = "leaked_target_copy",
    noise_std: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Menanamkan defect L1: Fitur yang disalin langsung dari target dengan sedikit derau."""
    df_mod = df.copy()
    rng = np.random.RandomState(seed)
    y = df_mod[target_col]
    if pd.api.types.is_numeric_dtype(y):
        df_mod[col_name] = y + rng.normal(0, noise_std, size=len(y))
    else:
        # Kategorikal
        df_mod[col_name] = y.copy()
    return df_mod


def seed_l2_cross_split_duplicates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    duplicate_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Menanamkan defect L2: Menyalin baris dari train_df ke test_df."""
    rng = np.random.RandomState(seed)
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    n_dups = int(len(test_copy) * duplicate_ratio)
    if n_dups > 0 and len(train_copy) >= n_dups:
        dup_indices = rng.choice(train_copy.index, size=n_dups, replace=False)
        dup_rows = train_copy.loc[dup_indices].copy()
        test_copy.iloc[:n_dups] = dup_rows.values

    return train_copy, test_copy


def seed_l5_temporal_leakage(
    n_samples: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Menanamkan defect L5: Data deret waktu dengan kolom tanggal yang diacak."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n_samples, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "x1": rng.normal(0, 1, n_samples),
        "x2": rng.uniform(10, 50, n_samples),
    })
    # Target memiliki tren waktu
    trend = np.linspace(0, 2, n_samples)
    df["target"] = (df["x1"] + trend + rng.normal(0, 0.5, n_samples) > 1.0).astype(int)
    return df


def seed_l6_group_leakage(
    n_samples: int = 600,
    n_groups: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Menanamkan defect L6: Kolom ID entitas berulang (grup) yang bocor ke data uji."""
    rng = np.random.RandomState(seed)
    groups = [f"user_{i:03d}" for i in range(n_groups)]
    user_col = rng.choice(groups, size=n_samples)

    # Setiap user memiliki baseline target tersendiri (efek acak kuat)
    user_effects = {g: rng.normal(0, 1.5) for g in groups}
    base_effect = np.array([user_effects[u] for u in user_col])

    x = rng.normal(0, 1, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-(x + base_effect)))
    target = (probs > 0.5).astype(int)

    return pd.DataFrame({
        "customer_id": user_col,
        "feature_x": x,
        "target": target,
    })


def seed_l8_index_leakage(
    df: pd.DataFrame,
    col_name: str = "customer_id",
    is_monotonic: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Menanamkan defect L8: Kolom ID atau indeks baris dengan kardinalitas unik 100%."""
    df_mod = df.copy()
    n = len(df_mod)
    if is_monotonic:
        df_mod[col_name] = [f"ID_{i:06d}" for i in range(n)]
    else:
        rng = np.random.RandomState(seed)
        df_mod[col_name] = [f"ID_{val}" for val in rng.permutation(n)]
    return df_mod


def generate_linear_recovery_dataset(
    n_samples: int = 400,
    n_features: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Membangkitkan data sintetis dengan struktur murni linear."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    # Logit murni linear
    logits = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] + rng.normal(0, 0.4, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["target"] = (probs > 0.5).astype(int)
    return df


def generate_nonlinear_recovery_dataset(
    n_samples: int = 400,
    n_features: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Membangkitkan data sintetis dengan interaksi non-linear kuat (XOR / polimorfik)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    # Non-linear interaction: XOR pattern between quadrant of x0 and x1, plus non-linear polynomial in x2
    xor_signal = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    circle_signal = (X[:, 2] ** 2 + X[:, 3] ** 2 > 1.5).astype(float)
    logits = 2.5 * (xor_signal - 0.5) + 1.8 * (circle_signal - 0.5) + rng.normal(0, 0.3, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["target"] = (probs > 0.5).astype(int)
    return df


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Menghitung titik estimasi dan interval kepercayaan Wilson 95%."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996  # 95%
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p, lower, upper

