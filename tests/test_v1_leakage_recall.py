# tests/test_v1_leakage_recall.py
"""
Verifikasi V1: Recall detektor kebocoran data pada defect tersemai (PRD 10.1 & Tabel 10.3).
Target [T]: Recall >= 80% per jenis kebocoran (L1, L2, L5, L6) dengan interval Wilson 95%.
"""

import pytest
import numpy as np
import pandas as pd
from shadow_core.leakage.l1_target import detect_l1_target_leakage
from shadow_core.leakage.l2_duplicates import detect_l2_cross_split_duplicates
from shadow_core.leakage.l5_temporal import detect_l5_temporal_leakage
from shadow_core.leakage.l6_group import detect_l6_group_leakage
from shadow_eval.seed_defects import (
    generate_clean_dataset,
    seed_l1_target_copy,
    seed_l2_cross_split_duplicates,
    seed_l5_temporal_leakage,
    seed_l6_group_leakage,
    wilson_score_interval,
)


def test_v1_l1_target_recall():
    """V1 — L1 Target derivation leakage recall (>= 80%)."""
    n_cases = 10
    detected = 0

    for seed in range(n_cases):
        base_df = generate_clean_dataset(n_samples=500, task="binary", seed=seed + 100)
        leaked_df = seed_l1_target_copy(base_df, target_col="target", seed=seed)
        rep = detect_l1_target_leakage(leaked_df, target_col="target")
        if not rep.passed and rep.findings:
            detected += 1

    p, low, high = wilson_score_interval(detected, n_cases)
    print(f"\nV1 L1 Recall: {detected}/{n_cases} ({p:.1%}) | Wilson 95%: [{low:.1%}, {high:.1%}]")
    assert p >= 0.80, f"Recall L1 {p:.1%} di bawah target 80%!"


def test_v1_l2_duplicates_recall():
    """V1 — L2 Cross-split duplicates recall (>= 80%)."""
    n_cases = 10
    detected = 0

    for seed in range(n_cases):
        clean_train = generate_clean_dataset(n_samples=400, task="binary", seed=seed + 200)
        clean_test = generate_clean_dataset(n_samples=100, task="binary", seed=seed + 300)
        train_mod, test_mod = seed_l2_cross_split_duplicates(clean_train, clean_test, duplicate_ratio=0.20, seed=seed)
        rep = detect_l2_cross_split_duplicates(train_mod, test_mod)
        if not rep.passed and rep.exact_duplicate_count > 0:
            detected += 1

    p, low, high = wilson_score_interval(detected, n_cases)
    print(f"\nV1 L2 Recall: {detected}/{n_cases} ({p:.1%}) | Wilson 95%: [{low:.1%}, {high:.1%}]")
    assert p >= 0.80, f"Recall L2 {p:.1%} di bawah target 80%!"


def test_v1_l5_temporal_recall():
    """V1 — L5 Temporal leakage recall (>= 80%)."""
    n_cases = 10
    detected = 0

    for seed in range(n_cases):
        df_temp = seed_l5_temporal_leakage(n_samples=500, seed=seed + 400)
        # Kasus defect: data berwaktu di-split dengan KFold acak
        rep = detect_l5_temporal_leakage(df_temp, target_col="target", split_strategy_used="StratifiedKFold")
        if not rep.passed and rep.is_blocker:
            detected += 1

    p, low, high = wilson_score_interval(detected, n_cases)
    print(f"\nV1 L5 Recall: {detected}/{n_cases} ({p:.1%}) | Wilson 95%: [{low:.1%}, {high:.1%}]")
    assert p >= 0.80, f"Recall L5 {p:.1%} di bawah target 80%!"


def test_v1_l6_group_recall():
    """V1 — L6 Group leakage recall (>= 80%)."""
    n_cases = 10
    detected = 0

    for seed in range(n_cases):
        df_group = seed_l6_group_leakage(n_samples=500, seed=seed + 500)
        # Simulasi split acak yang membocorkan grup ke kedua split
        train_df = df_group.iloc[:350]
        test_df = df_group.iloc[250:]  # Sengaja overlap baris 250..350
        rep = detect_l6_group_leakage(train_df, test_df, group_col="customer_id")
        if not rep.passed and rep.overlapping_groups_count > 0:
            detected += 1

    p, low, high = wilson_score_interval(detected, n_cases)
    print(f"\nV1 L6 Recall: {detected}/{n_cases} ({p:.1%}) | Wilson 95%: [{low:.1%}, {high:.1%}]")
    assert p >= 0.80, f"Recall L6 {p:.1%} di bawah target 80%!"
