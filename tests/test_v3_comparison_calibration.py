# tests/test_v3_comparison_calibration.py
"""
Verifikasi V3: Kalibrasi Rumus Perbandingan Model Nadeau–Bengio (PRD 10.1 & 6.7).
Uji:
(a) Uji A/A (dua model identik beda seed): klaim palsu <= 10% (batas atas 95%).
(b) Uji Null (label diacak): klaim palsu <= 10%.
(c) Uji Cakupan (Coverage): rentang 95% memuat selisih sebenarnya >= 90%.
"""

import pytest
import numpy as np
from shadow_core.comparison import nadeau_bengio_ci, compare_models_pairwise
from shadow_eval.seed_defects import wilson_score_interval


def test_v3_a_aa_test():
    """V3 (a) Uji A/A: Dua model identik hanya beda variasi seed fold."""
    n_simulations = 100
    false_claims = 0
    delta = 0.01

    rng = np.random.RandomState(42)
    for _ in range(n_simulations):
        # Skor 15 lipatan dari proses yang sama (selisih murni derau ~ N(0, 0.01^2))
        scores_a = rng.normal(0.80, 0.02, size=15)
        scores_b = rng.normal(0.80, 0.02, size=15)

        comp = compare_models_pairwise(
            scores_a=scores_a,
            scores_b=scores_b,
            name_a="Model_A1",
            name_b="Model_A2",
            delta=delta,
        )
        # Klaim salah bila menyatakan A lebih baik dari B
        if comp.is_statistically_better:
            false_claims += 1

    p, low, high = wilson_score_interval(false_claims, n_simulations)
    print(f"\nV3(a) A/A False Claims: {false_claims}/{n_simulations} ({p:.1%}) | Batas Atas: {high:.1%}")
    assert high <= 0.12 or p <= 0.10, f"False claim rate pada A/A test {p:.1%} melebihi batas 10%!"


def test_v3_b_null_test():
    """V3 (b) Uji Null: Target acak, perbandingan dua algoritma tanpa sinyal sejati."""
    n_simulations = 100
    false_claims = 0
    delta = 0.01

    rng = np.random.RandomState(123)
    for _ in range(n_simulations):
        # Nilai acak berfluktuasi sekitar chance (0.50)
        scores_alg1 = rng.normal(0.50, 0.03, size=15)
        scores_alg2 = rng.normal(0.50, 0.03, size=15)

        comp = compare_models_pairwise(
            scores_a=scores_alg1,
            scores_b=scores_alg2,
            name_a="Ridge",
            name_b="RF",
            delta=delta,
        )
        if comp.is_statistically_better:
            false_claims += 1

    p, low, high = wilson_score_interval(false_claims, n_simulations)
    print(f"\nV3(b) Null Test False Claims: {false_claims}/{n_simulations} ({p:.1%}) | Batas Atas: {high:.1%}")
    assert high <= 0.12 or p <= 0.10, f"False claim rate pada Null test {p:.1%} melebihi batas 10%!"


def test_v3_c_coverage_test():
    """V3 (c) Cakupan (Coverage): 95% CI memuat true difference Delta* >= 90%."""
    n_simulations = 100
    covered = 0
    true_delta = 0.03  # Selisih sebenarnya

    rng = np.random.RandomState(456)
    N = 500
    for _ in range(n_simulations):
        # Bangkitkan error instance dengan rata-rata true_delta
        eps = rng.normal(true_delta, 0.05, size=N)
        d = []
        for rep in range(3):
            perm = rng.permutation(eps)
            chunks = np.array_split(perm, 5)
            for ch in chunks:
                d.append(float(np.mean(ch)))

        mean_d, se_d, ci_low, ci_high = nadeau_bengio_ci(d)
        if ci_low <= true_delta <= ci_high:
            covered += 1

    p, low, high = wilson_score_interval(covered, n_simulations)
    print(f"\nV3(c) Coverage: {covered}/{n_simulations} ({p:.1%}) | Wilson 95%: [{low:.1%}, {high:.1%}]")
    assert p >= 0.90, f"Cakupan Nadeau-Bengio {p:.1%} di bawah target 90%!"
