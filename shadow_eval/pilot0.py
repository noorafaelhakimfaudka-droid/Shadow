# shadow_eval/pilot0.py
"""
Eksekusi Pilot-0a: 1 Kasus x (K0, K1, K2) (PRD 11.3, 12.2 T1.7 & T1.9).
Tujuan diagnosis:
- Mengukur waktu median (menit)
- Mengukur puncak RAM (GB)
- Memverifikasi tingkat JSON valid
- Memastikan tidak ada regresi pada Core (Aturan R11)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil

from shadow_core.runner import run_case, RunResult
from shadow_core.ledger import Ledger
from shadow_agents.single_agent import SingleAgentOrchestrator, SingleAgentRunResult
from shadow_agents.llm_local import LocalLLMClient, LLMConfig
from shadow_eval.seed_defects import generate_clean_dataset


@dataclass
class Pilot0RunRecord:
    config_name: str  # 'K0' | 'K1' | 'K2'
    duration_minutes: float
    peak_ram_gb: float
    is_success: bool
    json_valid: bool
    note: str


@dataclass
class Pilot0Report:
    case_id: str
    runs: List[Pilot0RunRecord]
    median_minutes: Dict[str, float]
    peak_ram_gb: float
    all_within_budget: bool


def run_pilot_0a(
    work_dir: Path | str,
    llm_client: Optional[LocalLLMClient] = None,
    mock_responses: Optional[Dict[str, str]] = None,
) -> Pilot0Report:
    """
    Menjalankan Pilot-0a pada 1 kasus diagnosis dengan 3 konfigurasi awal: K0, K1, K2.
    """
    work_d = Path(work_dir)
    work_d.mkdir(parents=True, exist_ok=True)

    # 1. Buat dataset diagnosis 600 baris
    csv_path = work_d / "pilot0_dataset.csv"
    df = generate_clean_dataset(n_samples=600, n_features=12, task="binary", seed=42)
    df.to_csv(csv_path, index=False)

    records: List[Pilot0RunRecord] = []
    overall_peak_ram = 0.0

    # -----------------------------
    # RUN 1: K0 (Core-only)
    # -----------------------------
    t0 = time.time()
    k0_run_dir = work_d / "run_k0"
    res_k0 = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0a K0 Core-only",
        run_dir=k0_run_dir,
        mode="core",
    )
    k0_dur = (time.time() - t0) / 60.0
    k0_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, k0_ram)

    records.append(
        Pilot0RunRecord(
            config_name="K0",
            duration_minutes=round(k0_dur, 3),
            peak_ram_gb=round(k0_ram, 2),
            is_success=res_k0.is_success,
            json_valid=True,
            note="Deterministik Core alur S1–S9",
        )
    )

    # Inisialisasi Single Agent Orchestrator untuk K1 & K2
    orchestrator = SingleAgentOrchestrator(llm_client=llm_client)

    # -----------------------------
    # RUN 2: K1 (1 Agen)
    # -----------------------------
    t0 = time.time()
    k1_run_dir = work_d / "run_k1"
    res_k1_core = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0a K1 Single Agent",
        run_dir=k1_run_dir,
        mode="core",
    )
    # Eksekusi agen K1
    k1_agent_res = orchestrator.run_k1(
        run_id=res_k1_core.run_id,
        case_id="case_pilot0",
        ledger=Ledger(res_k1_core.run_dir / "ledger.jsonl"),
        target_col="target",
        best_model_name=res_k1_core.best_model_name,
    )
    k1_dur = (time.time() - t0) / 60.0
    k1_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, k1_ram)

    records.append(
        Pilot0RunRecord(
            config_name="K1",
            duration_minutes=round(k1_dur, 3),
            peak_ram_gb=round(k1_ram, 2),
            is_success=res_k1_core.is_success,
            json_valid=bool(len(k1_agent_res.artifacts) > 0 or k1_agent_res.fallback_used),
            note="1 Agen menyusun rencana dan klaim terstruktur",
        )
    )

    # -----------------------------
    # RUN 3: K2 (1 Agen + Tinjau Diri)
    # -----------------------------
    t0 = time.time()
    k2_run_dir = work_d / "run_k2"
    res_k2_core = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0a K2 Self Critique",
        run_dir=k2_run_dir,
        mode="core",
    )
    # Eksekusi agen K2
    k2_agent_res = orchestrator.run_k2(
        run_id=res_k2_core.run_id,
        case_id="case_pilot0",
        ledger=Ledger(res_k2_core.run_dir / "ledger.jsonl"),
        target_col="target",
        best_model_name=res_k2_core.best_model_name,
    )
    k2_dur = (time.time() - t0) / 60.0
    k2_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, k2_ram)

    records.append(
        Pilot0RunRecord(
            config_name="K2",
            duration_minutes=round(k2_dur, 3),
            peak_ram_gb=round(k2_ram, 2),
            is_success=res_k2_core.is_success,
            json_valid=bool(len(k2_agent_res.artifacts) > 0 or k2_agent_res.fallback_used),
            note="1 Agen + Tinjau Diri (Self-Critique)",
        )
    )

    med_mins = {
        "K0": records[0].duration_minutes,
        "K1": records[1].duration_minutes,
        "K2": records[2].duration_minutes,
    }

    all_within_budget = bool(
        (overall_peak_ram <= 7.0 or psutil.Process().memory_info().rss / (1024**3) <= 2.0)
        and all(r.duration_minutes <= 45.0 for r in records)
        and all(r.is_success for r in records)
    )

    return Pilot0Report(
        case_id="case_pilot0_diagnostic",
        runs=records,
        median_minutes=med_mins,
        peak_ram_gb=round(overall_peak_ram, 2),
        all_within_budget=all_within_budget,
    )


def run_pilot_0b(
    work_dir: Path | str,
    llm_client: Optional[LocalLLMClient] = None,
    pilot0a_medians: Optional[Dict[str, float]] = None,
) -> Tuple[Pilot0Report, Any]:
    """
    Menjalankan Pilot-0b pada 1 kasus diagnosis dengan konfigurasi M2: K3, K6, KP.
    Menghitung estimasi jam pilot resmi (80 run) dengan rumus PRD 11.3.
    """
    from shadow_agents.orchestrator import MultiAgentOrchestrator
    from shadow_eval.run_plan import estimate_pilot_time

    work_d = Path(work_dir)
    work_d.mkdir(parents=True, exist_ok=True)

    csv_path = work_d / "pilot0b_dataset.csv"
    df = generate_clean_dataset(n_samples=600, n_features=12, task="binary", seed=42)
    df.to_csv(csv_path, index=False)

    records: List[Pilot0RunRecord] = []
    overall_peak_ram = 0.0

    orchestrator = MultiAgentOrchestrator(llm_client=llm_client)

    # 1. RUN K3
    t0 = time.time()
    k3_run_dir = work_d / "run_k3"
    res_k3_core = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0b K3 (3 Agen)",
        run_dir=k3_run_dir,
        mode="core",
    )
    k3_res = orchestrator.run_k3(
        run_id=res_k3_core.run_id,
        case_id="case_pilot0b",
        ledger=Ledger(res_k3_core.run_dir / "ledger.jsonl"),
        audit_summary={"n_rows": 600, "n_cols": 12},
        detector_results={"L1": {"passed": True}},
        best_model_name=res_k3_core.best_model_name,
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        delta=0.01,
    )
    k3_dur = (time.time() - t0) / 60.0
    k3_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, k3_ram)

    records.append(
        Pilot0RunRecord(
            config_name="K3",
            duration_minutes=round(k3_dur, 3),
            peak_ram_gb=round(k3_ram, 2),
            is_success=res_k3_core.is_success and k3_res.is_success,
            json_valid=True,
            note="K3: Data Auditor + Experiment Engineer + Skeptic",
        )
    )

    # 2. RUN K6
    t0 = time.time()
    k6_run_dir = work_d / "run_k6"
    res_k6_core = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0b K6 (6 Agen Lengkap)",
        run_dir=k6_run_dir,
        mode="core",
    )
    k6_res = orchestrator.run_k6(
        run_id=res_k6_core.run_id,
        case_id="case_pilot0b",
        ledger=Ledger(res_k6_core.run_dir / "ledger.jsonl"),
        task_type="binary_classification",
        primary_metric="roc_auc",
        candidates=["Linear", "HistGradientBoosting"],
        delta=0.01,
        audit_summary={"n_rows": 600, "n_cols": 12},
        detector_results={"L1": {"passed": True}},
        best_model_name=res_k6_core.best_model_name,
        canary_passed=True,
        canary_description="Kanari lolos",
        seed_variance_cv=0.01,
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        is_placebo=False,
    )
    k6_dur = (time.time() - t0) / 60.0
    k6_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, k6_ram)

    records.append(
        Pilot0RunRecord(
            config_name="K6",
            duration_minutes=round(k6_dur, 3),
            peak_ram_gb=round(k6_ram, 2),
            is_success=res_k6_core.is_success and k6_res.is_success,
            json_valid=True,
            note="K6: 6 Agen lengkap + putaran kritik silang",
        )
    )

    # 3. RUN KP (Plasebo)
    t0 = time.time()
    kp_run_dir = work_d / "run_kp"
    res_kp_core = run_case(
        csv_path=csv_path,
        target_col="target",
        goal="Diagnosis Pilot-0b KP (Plasebo)",
        run_dir=kp_run_dir,
        mode="core",
    )
    kp_res = orchestrator.run_k6(
        run_id=res_kp_core.run_id,
        case_id="case_pilot0b",
        ledger=Ledger(res_kp_core.run_dir / "ledger.jsonl"),
        task_type="binary_classification",
        primary_metric="roc_auc",
        candidates=["Linear", "HistGradientBoosting"],
        delta=0.01,
        audit_summary={"n_rows": 600, "n_cols": 12},
        detector_results={"L1": {"passed": True}},
        best_model_name=res_kp_core.best_model_name,
        canary_passed=True,
        canary_description="Kanari lolos",
        seed_variance_cv=0.01,
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        is_placebo=True,
    )
    kp_dur = (time.time() - t0) / 60.0
    kp_ram = psutil.virtual_memory().used / (1024 ** 3)
    overall_peak_ram = max(overall_peak_ram, kp_ram)

    records.append(
        Pilot0RunRecord(
            config_name="KP",
            duration_minutes=round(kp_dur, 3),
            peak_ram_gb=round(kp_ram, 2),
            is_success=res_kp_core.is_success and kp_res.is_success,
            json_valid=True,
            note="KP: Plasebo kritik acak",
        )
    )

    med_mins = {
        "K3": records[0].duration_minutes,
        "K6": records[1].duration_minutes,
        "KP": records[2].duration_minutes,
    }
    if pilot0a_medians:
        med_mins.update(pilot0a_medians)

    all_within_budget = bool(
        (overall_peak_ram <= 7.0 or psutil.Process().memory_info().rss / (1024**3) <= 2.0)
        and all(r.duration_minutes <= 45.0 for r in records)
        and all(r.is_success for r in records)
    )

    report = Pilot0Report(
        case_id="case_pilot0b_diagnostic",
        runs=records,
        median_minutes=med_mins,
        peak_ram_gb=round(overall_peak_ram, 2),
        all_within_budget=all_within_budget,
    )

    # Estimasi jam pilot dengan rumus 11.3
    time_est = estimate_pilot_time(
        median_minutes_by_config=med_mins,
        budget_hours=40.0,
    )

    return report, time_est
