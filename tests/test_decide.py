# tests/test_decide.py
"""
10 Uji Unit Wajib untuk 'pilih_default' (PRD Tabel 11.5 & AC-22).
Memastikan fungsi kode deterministik memilih konfigurasi secara adil dan tepat.
"""

import pytest
from shadow_eval.decide import (
    KonfigurasiHasil,
    EvaluasiHasil,
    pilih_default,
    Keputusan,
)


def get_base_k0():
    return KonfigurasiHasil(
        recall_mean=0.50,
        recall_ci_lower=0.45,
        precision_mean=0.90,
        median_time_minutes=3.0,
        failure_rate=0.0,
        false_positives_per_run=0.0,
    )


def test_scenario_01_no_value_added():
    """1. Tidak ada konfigurasi yang memberi nilai tambah -> default = K0."""
    k0 = get_base_k0()
    k1 = KonfigurasiHasil(recall_mean=0.52, recall_ci_lower=0.48, precision_mean=0.85, median_time_minutes=8.0, failure_rate=0.0, false_positives_per_run=0.5)
    k2 = KonfigurasiHasil(recall_mean=0.55, recall_ci_lower=0.49, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.0, false_positives_per_run=0.5)
    
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K1": k1, "K2": k2},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K0"


def test_scenario_02_k2_passes():
    """2. K2 +0.20 recall (batas bawah > 0), presisi 0.85, waktu normal -> default = K2."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(
        recall_mean=0.70,  # +0.20 vs K0
        recall_ci_lower=0.52,  # > 0.50
        precision_mean=0.85,
        median_time_minutes=14.0,
        failure_rate=0.01,
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K2"


def test_scenario_03_k6_passes_with_reasoning():
    """3. K6 lolos vs K2, waktu 2.5x K2, K6 > KP -> default = K6, klaim_penalaran = True."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.0, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.88,  # +0.18 vs K2
        recall_ci_lower=0.72,  # > 0.70
        precision_mean=0.85,
        median_time_minutes=35.0,  # 2.5x K2 (35 <= 42)
        failure_rate=0.02,
        false_positives_per_run=0.8,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
        k6_vs_kp_recall_ci_lower=0.05,  # K6 > KP
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K6"
    assert kep.klaim_penalaran is True


def test_scenario_04_k6_too_slow():
    """4. K6 lolos recall tetapi waktu 3.5x K2 -> default tetap K2."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=10.0, failure_rate=0.0, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.88,
        recall_ci_lower=0.72,
        precision_mean=0.85,
        median_time_minutes=35.0,  # 3.5x K2 (> 3x K2)
        failure_rate=0.0,
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K2"


def test_scenario_05_k6_ci_crosses_zero():
    """5. K6 +0.20 recall tetapi batas bawah <= 0 -> default tetap K2."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.0, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.90,
        recall_ci_lower=0.68,  # diff_lower = 0.68 - 0.70 = -0.02 <= 0
        precision_mean=0.85,
        median_time_minutes=25.0,
        failure_rate=0.0,
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K2"


def test_scenario_06_k6_low_precision():
    """6. K6 recall naik tetapi presisi 0.70 -> default tetap K2."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.0, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.90,
        recall_ci_lower=0.75,
        precision_mean=0.70,  # < 0.80
        median_time_minutes=25.0,
        failure_rate=0.0,
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K2"


def test_scenario_07_k6_passes_but_kp_not_beaten():
    """7. K6 lolos, tetapi K6 - KP batas bawah <= 0 -> default = K6, klaim_penalaran = False."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.0, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.88,
        recall_ci_lower=0.72,
        precision_mean=0.85,
        median_time_minutes=30.0,
        failure_rate=0.0,
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
        k6_vs_kp_recall_ci_lower=-0.02,  # K6 tidak mengungguli KP secara statistik
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K6"
    assert kep.klaim_penalaran is False


def test_scenario_08_only_10_heldout_cases():
    """8. Hanya 10 kasus simpanan -> status = 'indikatif'."""
    k0 = get_base_k0()
    hasil = EvaluasiHasil(
        configs={"K0": k0},
        n_kasus_simpanan=10,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.status == "indikatif"


def test_scenario_09_k6_high_failure_rate():
    """9. Tingkat gagal K6 lebih tinggi dari K2 lebih dari 0.05 -> default tetap K2."""
    k0 = get_base_k0()
    k2 = KonfigurasiHasil(recall_mean=0.70, recall_ci_lower=0.55, precision_mean=0.85, median_time_minutes=14.0, failure_rate=0.01, false_positives_per_run=0.5)
    k6 = KonfigurasiHasil(
        recall_mean=0.88,
        recall_ci_lower=0.72,
        precision_mean=0.85,
        median_time_minutes=30.0,
        failure_rate=0.08,  # 0.08 > 0.01 + 0.05
        false_positives_per_run=0.5,
    )
    hasil = EvaluasiHasil(
        configs={"K0": k0, "K2": k2, "K6": k6},
        n_kasus_simpanan=35,
        protokol_penuh=True,
    )
    kep = pilih_default(hasil)
    assert kep.default_config == "K2"


def test_scenario_10_invalid_or_small_input():
    """10. Input kosong / kasus < 5 -> galat terstruktur ValueError, tidak ada keputusan."""
    hasil_small = EvaluasiHasil(configs={"K0": get_base_k0()}, n_kasus_simpanan=3)
    with pytest.raises(ValueError):
        pilih_default(hasil_small)

    hasil_empty = EvaluasiHasil(configs={}, n_kasus_simpanan=10)
    with pytest.raises(ValueError):
        pilih_default(hasil_empty)
