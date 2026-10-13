# tests/test_pilot_runner.py
"""
Uji Harness Pilot 80-Run & Evaluator Go/No-Go (PRD 11.3, 11.4, 12.3 T2.7).
"""

from pathlib import Path
import pytest
from shadow_agents.llm_local import LocalLLMClient
from shadow_eval.pilot_runner import (
    prepare_standard_pilot_cases,
    run_full_pilot,
    FullPilotReport,
)


def test_prepare_standard_pilot_cases(tmp_path: Path):
    """Memverifikasi penyiapan 5 kasus pilot baku (clean, L1, L5, L6, L8)."""
    cases = prepare_standard_pilot_cases(tmp_path)
    assert len(cases) == 5

    case_ids = [c.case_id for c in cases]
    assert case_ids == ["case1_clean", "case2_l1", "case3_l5", "case4_l6", "case5_l8"]

    for c in cases:
        assert c.csv_path.exists()
        assert c.defects_yaml_path.exists()


def test_pilot_runner_execution_and_decision(tmp_path: Path):
    """
    Menjalankan simulasi pilot cepat pada kasus pilot.
    Memverifikasi seluruh konfigurasi (K0, K1, K2, K3, K6, KP) dievaluasi,
    dan pilih_default menghasilkan keputusan go/no-go berstatus 'indikatif'.
    """
    mock_llm = LocalLLMClient(
        mock_response='{"artifact_id": "a-001", "run_id": "r-001", "agent": "auditor", "seq": 1, "jenis": "temuan", "klaim": "Deteksi anomali pada fitur.", "rujukan_bukti": ["trial:t-001"], "keyakinan": "tinggi", "alasan_keyakinan": "audit", "risiko_asumsi": "none"}'
    )

    # Siapkan 2 kasus untuk pengujian integrasi cepat
    all_cases = prepare_standard_pilot_cases(tmp_path / "pilot_prep")
    test_cases = all_cases[:2]  # Kasus clean dan L1

    report = run_full_pilot(
        work_dir=tmp_path / "pilot_run",
        cases=test_cases,
        llm_client=mock_llm,
        fast_mode=True,  # 1 ulangan per konfigurasi untuk uji unit
    )

    # 2 kasus x (K0 + K1 + K2 + K3 + K6 + KP) = 2 x 6 = 12 runs
    assert report.total_runs == 12
    assert report.completed_runs == 12
    assert len(report.metrics_by_config) == 6

    for cfg in ["K0", "K1", "K2", "K3", "K6", "KP"]:
        assert cfg in report.metrics_by_config
        metrics = report.metrics_by_config[cfg]
        assert "recall" in metrics
        assert "precision" in metrics
        assert "median_minutes" in metrics

    # PRD 11.5: Status pilot adalah 'indikatif' (karena kasus simpanan < 30)
    assert report.decision.status == "indikatif"
    assert report.decision.default_config in ["K0", "K1", "K2", "K3", "K6"]
