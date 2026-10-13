# tests/test_run_plan.py
"""
Uji Perhitungan Rencana Run & Tangga Pemangkasan Waktu (PRD 11.3, 12.2 T1.6).
"""

import pytest
from shadow_eval.run_plan import (
    calculate_run_counts,
    estimate_pilot_time,
    estimate_benchmark_m3_time,
)


def test_standard_pilot_run_counts():
    """Memverifikasi hitungan run baku PRD 11.3: 80 pilot runs + 6 Pilot-0 = 86 total."""
    counts = calculate_run_counts(n_cases_pilot=5, repeats=3)
    assert counts.pilot0_runs == 6
    assert counts.runs_by_config["K0"] == 5
    assert counts.runs_by_config["K1"] == 15
    assert counts.runs_by_config["K2"] == 15
    assert counts.runs_by_config["K3"] == 15
    assert counts.runs_by_config["K6"] == 15
    assert counts.runs_by_config["KP"] == 15
    assert counts.pilot_runs == 80
    assert counts.total_pilot_runs == 86


def test_estimate_pilot_time_within_budget():
    """Memverifikasi estimasi waktu pilot dari contoh PRD 11.3 (30.25 jam <= 40 jam)."""
    medians = {
        "K0": 3.0,
        "K1": 8.0,
        "K2": 14.0,
        "K3": 22.0,
        "K6": 38.0,
        "KP": 38.0,
    }
    res = estimate_pilot_time(medians, budget_hours=40.0)
    assert res.within_budget is True
    assert res.total_hours == 30.25
    assert res.repeats == 3
    assert len(res.pruned_steps) == 0
    assert "K3" in res.active_configs
    assert "K1" in res.active_configs


def test_estimate_pilot_time_pruning_ladder():
    """
    Memverifikasi tangga pemangkasan PRD 11.3 saat waktu melebihi anggaran:
    1. repeats 3 -> 2 -> 1
    2. buang K3
    3. buang K1
    * K0, K2, K6, KP tidak pernah dibuang.
    """
    # Waktu median sangat besar, melebihi 40 jam
    heavy_medians = {
        "K0": 5.0,
        "K1": 30.0,
        "K2": 40.0,
        "K3": 60.0,
        "K6": 90.0,
        "KP": 90.0,
    }
    # Tanpa pangkas: 5*3*(30+40+60+90+90)/60 = 15*310/60 = 77.5 jam + 0.4 jam = 77.9 jam > 40 jam
    res = estimate_pilot_time(heavy_medians, budget_hours=40.0)

    # Harus memicu tangga pemangkasan
    assert len(res.pruned_steps) > 0
    # K0, K2, K6, KP tidak pernah dibuang
    for required in ["K0", "K2", "K6", "KP"]:
        assert required in res.active_configs


def test_estimate_benchmark_m3_time():
    """Memverifikasi estimasi waktu benchmark M3 dan pemangkasan ulangan 3 -> 2."""
    medians = {
        "K0": 3.0,
        "K1": 8.0,
        "K2": 14.0,
        "K3": 22.0,
        "K6": 38.0,
        "KP": 38.0,
    }
    # Kasus N=10, R=3 -> 60 jam (muat dalam budget 120 jam)
    res_10 = estimate_benchmark_m3_time(n_held_out_cases=10, median_minutes_by_config=medians, budget_hours=120.0)
    assert res_10.within_budget is True
    assert res_10.repeats == 3

    # Kasus N=30, jika waktu median besar sehingga > 120 jam -> ulangan turun ke 2
    heavy_medians = {
        "K0": 5.0,
        "K1": 20.0,
        "K2": 30.0,
        "K3": 40.0,
        "K6": 60.0,
        "KP": 60.0,
    }
    res_30 = estimate_benchmark_m3_time(n_held_out_cases=30, median_minutes_by_config=heavy_medians, budget_hours=120.0)
    assert res_30.repeats == 2
    assert "M3 Pangkas ulangan: 3 -> 2" in res_30.pruned_steps
