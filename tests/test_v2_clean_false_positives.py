# tests/test_v2_clean_false_positives.py
"""
Verifikasi V2: Blocker salah (False Positive) pada kasus bersih (PRD 10.1).
Target [T]: Blocker salah <= 10% pada >= 20 kasus bersih; batas atas Wilson dilaporkan.
"""

import pytest
import numpy as np
import pandas as pd
from shadow_core.audit import audit_dataset
from shadow_core.leakage.l1_target import detect_l1_target_leakage
from shadow_eval.seed_defects import generate_clean_dataset, wilson_score_interval


def test_v2_clean_cases_false_positive_rate():
    n_cases = 20
    false_blockers = 0

    for seed in range(n_cases):
        task = "binary" if seed % 2 == 0 else "regression"
        clean_df = generate_clean_dataset(n_samples=500, n_features=8, task=task, seed=seed + 1000)

        # Audit S2
        audit_rep = audit_dataset(clean_df, target_col="target")
        # Detektor L1
        l1_rep = detect_l1_target_leakage(clean_df, target_col="target")

        # Jika ada blocker yang muncul pada data bersih, hitung sebagai false positive
        has_false_blocker = audit_rep.has_blocker or (not l1_rep.passed and any(f.is_blocker for f in l1_rep.findings))
        if has_false_blocker:
            false_blockers += 1

    p, low, high = wilson_score_interval(false_blockers, n_cases)
    print(f"\nV2 False Positive Rate: {false_blockers}/{n_cases} ({p:.1%}) | Batas Atas Wilson 95%: {high:.1%}")

    # Target: batas atas <= 15% atau p <= 10%
    assert p <= 0.10, f"False positive rate {p:.1%} melebihi batas target 10%!"
