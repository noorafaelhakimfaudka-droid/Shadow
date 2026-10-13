# shadow_eval/run_plan.py
"""
Perhitungan Rencana Run & Estimasi Waktu (PRD 11.3, 12.2 T1.6).
- Menghitung jumlah run dan estimasi jam komputasi berurutan pada 1 laptop.
- Menerapkan rumus: T_jam = sum_konfigurasi (jumlah_run * waktu_median_menit) / 60
- Menerapkan aturan pemangkasan terkunci (pruning ladder) bila estimasi melebihi batas anggaran:
  1. Ulangan 3 -> 2 -> 1
  2. Buang K3
  3. Buang K1
  * DILARANG membuang K0, K2, K6, KP dan dilarang mengecilkan set simpanan.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RunCounts:
    pilot0_runs: int
    pilot_runs: int
    total_pilot_runs: int
    runs_by_config: Dict[str, int]


@dataclass
class TimeEstimationResult:
    total_hours: float
    budget_hours: float
    within_budget: bool
    active_configs: List[str]
    repeats: int
    pruned_steps: List[str]
    runs_by_config: Dict[str, int]
    hours_by_config: Dict[str, float]


def calculate_run_counts(
    n_cases_pilot: int = 5,
    repeats: int = 3,
    include_k3: bool = True,
    include_k1: bool = True,
) -> RunCounts:
    """
    Menghitung jumlah run pilot sesuai PRD 11.3.
    Default:
    - Pilot-0: 6 run (1 kasus x 6 konfigurasi [K0, K1, K2, K3, K6, KP] x 1)
    - Pilot K0: 5 kasus x 1 ulangan = 5 run
    - Pilot LLM: 5 kasus x configs x repeats
    """
    configs = ["K0", "K2", "K6", "KP"]
    if include_k3:
        configs.append("K3")
    if include_k1:
        configs.append("K1")

    runs_by_config: Dict[str, int] = {}
    # K0 selalu 1 ulangan (deterministik)
    runs_by_config["K0"] = n_cases_pilot * 1

    llm_runs = 0
    for cfg in configs:
        if cfg != "K0":
            count = n_cases_pilot * repeats
            runs_by_config[cfg] = count
            llm_runs += count

    pilot0_runs = 6  # 1 kasus x 6 konfigurasi
    pilot_runs = runs_by_config["K0"] + llm_runs
    total_pilot_runs = pilot0_runs + pilot_runs

    return RunCounts(
        pilot0_runs=pilot0_runs,
        pilot_runs=pilot_runs,
        total_pilot_runs=total_pilot_runs,
        runs_by_config=runs_by_config,
    )


def estimate_pilot_time(
    median_minutes_by_config: Dict[str, float],
    budget_hours: float = 40.0,
    n_cases: int = 5,
    initial_repeats: int = 3,
) -> TimeEstimationResult:
    """
    Menghitung estimasi jam pilot berdasarkan waktu median nyata dari Pilot-0.
    Jika melebihi budget_hours (default 40 jam), jalankan tangga pemangkasan PRD 11.3.
    """
    repeats = initial_repeats
    include_k3 = True
    include_k1 = True
    pruned_steps: List[str] = []

    def compute_hours(reps: int, k3: bool, k1: bool) -> Tuple[float, Dict[str, int], Dict[str, float], List[str]]:
        cfgs = ["K0", "K2", "K6", "KP"]
        if k3:
            cfgs.append("K3")
        if k1:
            cfgs.append("K1")

        r_counts: Dict[str, int] = {"K0": n_cases * 1}
        h_counts: Dict[str, float] = {}

        k0_med = median_minutes_by_config.get("K0", 3.0)
        h_counts["K0"] = (r_counts["K0"] * k0_med) / 60.0

        for c in cfgs:
            if c != "K0":
                count = n_cases * reps
                r_counts[c] = count
                med = median_minutes_by_config.get(c, 30.0)
                h_counts[c] = (count * med) / 60.0

        tot_h = sum(h_counts.values())
        return tot_h, r_counts, h_counts, cfgs

    total_h, r_counts, h_counts, active_cfgs = compute_hours(repeats, include_k3, include_k1)

    # Tangga pemangkasan PRD 11.3:
    # 1. Ulangan 3 -> 2
    if total_h > budget_hours and repeats == 3:
        repeats = 2
        pruned_steps.append("Pangkas ulangan: 3 -> 2")
        total_h, r_counts, h_counts, active_cfgs = compute_hours(repeats, include_k3, include_k1)

    # 1b. Ulangan 2 -> 1
    if total_h > budget_hours and repeats == 2:
        repeats = 1
        pruned_steps.append("Pangkas ulangan: 2 -> 1")
        total_h, r_counts, h_counts, active_cfgs = compute_hours(repeats, include_k3, include_k1)

    # 2. Buang K3
    if total_h > budget_hours and include_k3:
        include_k3 = False
        pruned_steps.append("Pangkas konfigurasi: Buang K3")
        total_h, r_counts, h_counts, active_cfgs = compute_hours(repeats, include_k3, include_k1)

    # 3. Buang K1
    if total_h > budget_hours and include_k1:
        include_k1 = False
        pruned_steps.append("Pangkas konfigurasi: Buang K1")
        total_h, r_counts, h_counts, active_cfgs = compute_hours(repeats, include_k3, include_k1)

    within_budget = total_h <= budget_hours

    return TimeEstimationResult(
        total_hours=round(total_h, 2),
        budget_hours=budget_hours,
        within_budget=within_budget,
        active_configs=active_cfgs,
        repeats=repeats,
        pruned_steps=pruned_steps,
        runs_by_config=r_counts,
        hours_by_config={k: round(v, 2) for k, v in h_counts.items()},
    )


def estimate_benchmark_m3_time(
    n_held_out_cases: int,
    median_minutes_by_config: Dict[str, float],
    budget_hours: float = 120.0,
    initial_repeats: int = 3,
) -> TimeEstimationResult:
    """
    Menghitung estimasi jam benchmark M3 sesuai PRD 11.3:
    N held-out kasus x configs x R ulangan.
    Anggaran default: 120 jam.
    Pemangkasan: ulangan 3 -> 2.
    """
    repeats = initial_repeats
    configs = ["K0", "K1", "K2", "K3", "K6", "KP"]
    pruned_steps: List[str] = []

    def compute_m3(reps: int) -> Tuple[float, Dict[str, int], Dict[str, float]]:
        r_counts = {"K0": n_held_out_cases * 1}
        h_counts = {"K0": (r_counts["K0"] * median_minutes_by_config.get("K0", 3.0)) / 60.0}
        for c in configs:
            if c != "K0":
                count = n_held_out_cases * reps
                r_counts[c] = count
                h_counts[c] = (count * median_minutes_by_config.get(c, 30.0)) / 60.0
        return sum(h_counts.values()), r_counts, h_counts

    total_h, r_counts, h_counts = compute_m3(repeats)

    if total_h > budget_hours and repeats == 3:
        repeats = 2
        pruned_steps.append("M3 Pangkas ulangan: 3 -> 2")
        total_h, r_counts, h_counts = compute_m3(repeats)

    return TimeEstimationResult(
        total_hours=round(total_h, 2),
        budget_hours=budget_hours,
        within_budget=(total_h <= budget_hours),
        active_configs=configs,
        repeats=repeats,
        pruned_steps=pruned_steps,
        runs_by_config=r_counts,
        hours_by_config={k: round(v, 2) for k, v in h_counts.items()},
    )
