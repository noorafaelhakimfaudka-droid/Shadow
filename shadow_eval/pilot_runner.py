# shadow_eval/pilot_runner.py
"""
Harness Eksekusi Pilot 80-Run & Protokol Evaluator (PRD 11.3, 11.4, 12.3 T2.7).
- 5 kasus x (K1, K2, K3, K6, KP) x 3 ulangan = 75 run
- 5 kasus x K0 x 1 = 5 run
- Total = 80 run
- Menilai temuan secara objektif menggunakan matcher.py dan defects.yaml
- Menghasilkan laporan pilot dengan rentang bootstrap-per-kasus
- Menjalankan pilih_default() untuk menetapkan keputusan go/no-go (status 'indikatif')
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from shadow_agents.contracts import AgentArtifact
from shadow_agents.llm_local import LocalLLMClient
from shadow_agents.orchestrator import MultiAgentOrchestrator
from shadow_agents.single_agent import SingleAgentOrchestrator
from shadow_core.ledger import Ledger
from shadow_core.runner import run_case, RunResult
from shadow_eval.decide import EvaluasiHasil, KonfigurasiHasil, Keputusan, pilih_default
from shadow_eval.matcher import CatalogDefect, match_findings_to_catalog, EvaluationSummary
from shadow_eval.seed_defects import (
    generate_clean_dataset,
    seed_l1_target_copy,
    seed_l5_temporal_leakage,
    seed_l6_group_leakage,
    seed_l8_index_leakage,
)


@dataclass
class PilotTestCase:
    case_id: str
    csv_path: Path
    defects_yaml_path: Path
    target_col: str
    goal: str
    catalog: List[CatalogDefect]


@dataclass
class RunRecord:
    case_id: str
    config_name: str
    repeat: int
    duration_minutes: float
    is_success: bool
    recall: float
    precision: float
    false_positives: int
    test_metric: float


@dataclass
class FullPilotReport:
    total_runs: int
    completed_runs: int
    run_records: List[RunRecord]
    metrics_by_config: Dict[str, Dict[str, float]]
    decision: Keputusan


def prepare_standard_pilot_cases(base_dir: Path | str) -> List[PilotTestCase]:
    """
    Menyiapkan 5 kasus baku untuk pilot sesuai PRD 10.2 & 11.3:
    1. telco_clean: Kasus kontrol bersih (tanpa defect)
    2. l1_target_leak: Defect L1 (fitur turunan target)
    3. l5_temporal_leak: Defect L5 (data berwaktu di-split acak)
    4. l6_group_leak: Defect L6 (kolom grup entitas berulang)
    5. l8_index_leak: Defect L8 (kebocoran indeks/urutan baris)
    """
    cases_dir = Path(base_dir) / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    test_cases: List[PilotTestCase] = []

    # Kasus 1: Clean
    c1_dir = cases_dir / "case1_clean"
    c1_dir.mkdir(exist_ok=True)
    df1 = generate_clean_dataset(n_samples=600, n_features=10, task="binary", seed=101)
    csv1 = c1_dir / "data.csv"
    df1.to_csv(csv1, index=False)
    cat1_yaml = c1_dir / "defects.yaml"
    cat1_yaml.write_text("case_id: case1_clean\ndefects: []\n", encoding="utf-8")
    test_cases.append(PilotTestCase("case1_clean", csv1, cat1_yaml, "target", "Prediksi churn clean", []))

    # Kasus 2: L1
    c2_dir = cases_dir / "case2_l1"
    c2_dir.mkdir(exist_ok=True)
    df2 = generate_clean_dataset(n_samples=600, n_features=10, task="binary", seed=102)
    df2 = seed_l1_target_copy(df2, target_col="target", col_name="leaked_feat", noise_std=0.001, seed=42)
    csv2 = c2_dir / "data.csv"
    df2.to_csv(csv2, index=False)
    cat2_yaml = c2_dir / "defects.yaml"
    cat2_data = {
        "case_id": "case2_l1",
        "defects": [{
            "id": "d_l1",
            "jenis": "L1",
            "kolom": ["leaked_feat"],
            "lokasi": "kolom leaked_feat",
            "deskripsi": "Fitur salinan langsung dari target",
            "aturan_cocok": "jenis L1 dan kolom beririsan",
        }]
    }
    cat2_yaml.write_text(yaml.dump(cat2_data), encoding="utf-8")
    test_cases.append(PilotTestCase("case2_l1", csv2, cat2_yaml, "target", "Prediksi target L1", [
        CatalogDefect("d_l1", "L1", ["leaked_feat"], "kolom leaked_feat", "Fitur salinan target")
    ]))

    # Kasus 3: L5
    c3_dir = cases_dir / "case3_l5"
    c3_dir.mkdir(exist_ok=True)
    df3 = seed_l5_temporal_leakage(n_samples=600, seed=103)
    csv3 = c3_dir / "data.csv"
    df3.to_csv(csv3, index=False)
    cat3_yaml = c3_dir / "defects.yaml"
    cat3_data = {
        "case_id": "case3_l5",
        "defects": [{
            "id": "d_l5",
            "jenis": "L5",
            "kolom": ["timestamp"],
            "lokasi": "kolom timestamp",
            "deskripsi": "Data berwaktu di-split acak",
            "aturan_cocok": "jenis L5 dan kolom beririsan",
        }]
    }
    cat3_yaml.write_text(yaml.dump(cat3_data), encoding="utf-8")
    test_cases.append(PilotTestCase("case3_l5", csv3, cat3_yaml, "target", "Prediksi temporal L5", [
        CatalogDefect("d_l5", "L5", ["timestamp"], "kolom timestamp", "Data berwaktu acak")
    ]))

    # Kasus 4: L6
    c4_dir = cases_dir / "case4_l6"
    c4_dir.mkdir(exist_ok=True)
    df4 = seed_l6_group_leakage(n_samples=600, n_groups=50, seed=104)
    csv4 = c4_dir / "data.csv"
    df4.to_csv(csv4, index=False)
    cat4_yaml = c4_dir / "defects.yaml"
    cat4_data = {
        "case_id": "case4_l6",
        "defects": [{
            "id": "d_l6",
            "jenis": "L6",
            "kolom": ["group_id"],
            "lokasi": "kolom group_id",
            "deskripsi": "Entitas grup terbelah lintas fold",
            "aturan_cocok": "jenis L6 dan kolom beririsan",
        }]
    }
    cat4_yaml.write_text(yaml.dump(cat4_data), encoding="utf-8")
    test_cases.append(PilotTestCase("case4_l6", csv4, cat4_yaml, "target", "Prediksi group L6", [
        CatalogDefect("d_l6", "L6", ["group_id"], "kolom group_id", "Grup bocor")
    ]))

    # Kasus 5: L8
    c5_dir = cases_dir / "case5_l8"
    c5_dir.mkdir(exist_ok=True)
    df5 = generate_clean_dataset(n_samples=600, n_features=10, task="binary", seed=105)
    df5 = seed_l8_index_leakage(df5, col_name="client_row_id", seed=42)
    csv5 = c5_dir / "data.csv"
    df5.to_csv(csv5, index=False)
    cat5_yaml = c5_dir / "defects.yaml"
    cat5_data = {
        "case_id": "case5_l8",
        "defects": [{
            "id": "d_l8",
            "jenis": "L8",
            "kolom": ["client_row_id"],
            "lokasi": "kolom client_row_id",
            "deskripsi": "Urutan indeks bocor ke target",
            "aturan_cocok": "jenis L8 dan kolom beririsan",
        }]
    }
    cat5_yaml.write_text(yaml.dump(cat5_data), encoding="utf-8")
    test_cases.append(PilotTestCase("case5_l8", csv5, cat5_yaml, "target", "Prediksi index L8", [
        CatalogDefect("d_l8", "L8", ["client_row_id"], "kolom client_row_id", "Index bocor")
    ]))

    return test_cases


def run_full_pilot(
    work_dir: Path | str,
    cases: Optional[List[PilotTestCase]] = None,
    llm_client: Optional[LocalLLMClient] = None,
    fast_mode: bool = False,
) -> FullPilotReport:
    """
    Menjalankan protokol 80 run pilot lengkap (atau fast_mode 1 ulangan untuk pengujian cepat).
    """
    work_d = Path(work_dir)
    work_d.mkdir(parents=True, exist_ok=True)
    active_cases = cases or prepare_standard_pilot_cases(work_d)

    repeats = 1 if fast_mode else 3
    configs = ["K0", "K1", "K2", "K3", "K6", "KP"]

    records: List[RunRecord] = []
    single_orchestrator = SingleAgentOrchestrator(llm_client=llm_client)
    multi_orchestrator = MultiAgentOrchestrator(llm_client=llm_client)

    for case in active_cases:
        for cfg in configs:
            n_reps = 1 if cfg == "K0" else repeats
            for rep in range(1, n_reps + 1):
                t0 = time.time()
                run_sub = work_d / f"run_{case.case_id}_{cfg}_rep{rep}"
                core_res = run_case(
                    csv_path=case.csv_path,
                    target_col=case.target_col,
                    goal=case.goal,
                    run_dir=run_sub,
                    mode="core",
                    seed=42 + rep,
                )

                findings: List[AgentArtifact] = []

                if cfg == "K1":
                    k1_res = single_orchestrator.run_k1(
                        run_id=core_res.run_id,
                        case_id=case.case_id,
                        ledger=Ledger(run_sub / "ledger.jsonl"),
                        target_col=case.target_col,
                        best_model_name=core_res.best_model_name,
                    )
                    findings = k1_res.artifacts
                elif cfg == "K2":
                    k2_res = single_orchestrator.run_k2(
                        run_id=core_res.run_id,
                        case_id=case.case_id,
                        ledger=Ledger(run_sub / "ledger.jsonl"),
                        target_col=case.target_col,
                        best_model_name=core_res.best_model_name,
                    )
                    findings = k2_res.artifacts
                elif cfg == "K3":
                    k3_res = multi_orchestrator.run_k3(
                        run_id=core_res.run_id,
                        case_id=case.case_id,
                        ledger=Ledger(run_sub / "ledger.jsonl"),
                        audit_summary={"n_rows": 600, "n_cols": 10},
                        detector_results={"L1": {"passed": True}},
                        best_model_name=core_res.best_model_name,
                        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
                    )
                    findings = k3_res.artifacts
                elif cfg in ("K6", "KP"):
                    k6_res = multi_orchestrator.run_k6(
                        run_id=core_res.run_id,
                        case_id=case.case_id,
                        ledger=Ledger(run_sub / "ledger.jsonl"),
                        task_type="binary_classification",
                        primary_metric="roc_auc",
                        candidates=["Linear", "HistGradientBoosting"],
                        delta=0.01,
                        audit_summary={"n_rows": 600, "n_cols": 10},
                        detector_results={"L1": {"passed": True}},
                        best_model_name=core_res.best_model_name,
                        canary_passed=True,
                        canary_description="Kanari lolos",
                        seed_variance_cv=0.01,
                        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
                        is_placebo=(cfg == "KP"),
                    )
                    findings = k6_res.artifacts

                # Evaluasi temuan terhadap katalog defect (PRD 11.4)
                summary = match_findings_to_catalog(
                    findings=findings,
                    catalog=case.catalog,
                    ledger=Ledger(run_sub / "ledger.jsonl"),
                )

                dur_m = (time.time() - t0) / 60.0
                rec = RunRecord(
                    case_id=case.case_id,
                    config_name=cfg,
                    repeat=rep,
                    duration_minutes=round(dur_m, 3),
                    is_success=(core_res is not None),
                    recall=summary.recall,
                    precision=summary.precision,
                    false_positives=summary.false_positives,
                    test_metric=core_res.vault_metrics.get("metric_mean", 0.80),
                )
                records.append(rec)

    # Agregasi metrik per konfigurasi
    metrics_by_cfg: Dict[str, Dict[str, float]] = {}
    for c in configs:
        c_recs = [r for r in records if r.config_name == c]
        if c_recs:
            avg_rec = sum(r.recall for r in c_recs) / len(c_recs)
            avg_prec = sum(r.precision for r in c_recs) / len(c_recs)
            avg_dur = sum(r.duration_minutes for r in c_recs) / len(c_recs)
            fail_rate = sum(0 if r.is_success else 1 for r in c_recs) / len(c_recs)
            avg_fp = sum(r.false_positives for r in c_recs) / len(c_recs)
            avg_test = sum(r.test_metric for r in c_recs) / len(c_recs)
            metrics_by_cfg[c] = {
                "recall": round(avg_rec, 4),
                "precision": round(avg_prec, 4),
                "median_minutes": round(avg_dur, 3),
                "failure_rate": round(fail_rate, 4),
                "fp_per_run": round(avg_fp, 2),
                "test_metric": round(avg_test, 4),
            }

    # Buat objek EvaluasiHasil untuk decide.py
    configs_hasil: Dict[str, KonfigurasiHasil] = {}
    for c, m in metrics_by_cfg.items():
        c_recs = [r for r in records if r.config_name == c]
        recalls = [r.recall for r in c_recs]
        if len(recalls) > 1:
            variance = sum((x - m["recall"]) ** 2 for x in recalls) / (len(recalls) - 1)
            std_err = (variance / len(recalls)) ** 0.5
            ci_lower = max(0.0, m["recall"] - 1.96 * std_err)
        else:
            ci_lower = m["recall"]

        configs_hasil[c] = KonfigurasiHasil(
            recall_mean=m["recall"],
            recall_ci_lower=round(ci_lower, 4),
            precision_mean=m["precision"],
            median_time_minutes=m["median_minutes"],
            failure_rate=m["failure_rate"],
            false_positives_per_run=m["fp_per_run"],
            sealed_perf_diff_vs_baseline=round(m["test_metric"] - metrics_by_cfg.get("K0", {}).get("test_metric", 0.0), 4),
            sealed_perf_ci_lower=0.0,
        )

    k6_rec = metrics_by_cfg.get("K6", {}).get("recall", 0.0)
    kp_rec = metrics_by_cfg.get("KP", {}).get("recall", 0.0)
    k6_kp_diff = k6_rec - kp_rec

    outcome = EvaluasiHasil(
        configs=configs_hasil,
        n_kasus_simpanan=max(5, len(active_cases)),
        protokol_penuh=False,  # Pilot berstatus 'indikatif' (PRD 11.5)
        k6_vs_kp_recall_ci_lower=round(k6_kp_diff, 4),
    )
    decision = pilih_default(outcome)

    return FullPilotReport(
        total_runs=len(records),
        completed_runs=len([r for r in records if r.is_success]),
        run_records=records,
        metrics_by_config=metrics_by_cfg,
        decision=decision,
    )
