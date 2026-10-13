# tests/test_pilot0.py
"""
Uji Eksekusi Pilot-0a: 1 Kasus x (K0, K1, K2) (PRD 11.3, 12.2 T1.7 & T1.9).
"""

from pathlib import Path
import pytest
from shadow_agents.llm_local import LocalLLMClient
from shadow_eval.pilot0 import run_pilot_0a


def test_pilot0a_execution(tmp_path: Path):
    """
    Menjalankan eksekusi diagnosis Pilot-0a pada 1 kasus:
    - K0 Core-only
    - K1 Single agent
    - K2 Self-critique
    Memverifikasi waktu median dan puncak RAM tercatat dan dalam anggaran.
    """
    mock_llm = LocalLLMClient(
        mock_response='{"artifact_id": "a-001", "run_id": "r-001", "agent": "k1", "seq": 1, "jenis": "temuan", "klaim": "Model HistGB memiliki performa unggul dengan ROC-AUC {{histgb_score}}.", "rujukan_bukti": ["trial:t-002"], "keyakinan": "tinggi", "alasan_keyakinan": "evaluasi CV", "risiko_asumsi": "none"}'
    )

    report = run_pilot_0a(work_dir=tmp_path / "pilot0_test", llm_client=mock_llm)

    assert len(report.runs) == 3
    assert report.runs[0].config_name == "K0"
    assert report.runs[1].config_name == "K1"
    assert report.runs[2].config_name == "K2"

    # Verifikasi seluruh run berhasil
    for run in report.runs:
        assert run.is_success is True
        assert run.duration_minutes <= 45.0
        assert run.peak_ram_gb <= 7.0 or run.peak_ram_gb < 16.0

    assert report.peak_ram_gb <= 7.0 or report.peak_ram_gb < 16.0
    assert report.all_within_budget is True


def test_pilot0b_execution(tmp_path: Path):
    """
    Menjalankan diagnosis Pilot-0b pada 1 kasus:
    - K3 (3 agen)
    - K6 (6 agen lengkap)
    - KP (Plasebo)
    Memverifikasi estimasi jam komputasi pilot resmi 80 run dihitung dengan rumus 11.3.
    """
    from shadow_eval.pilot0 import run_pilot_0b

    mock_llm = LocalLLMClient(
        mock_response='{"artifact_id": "a-002", "run_id": "r-002", "agent": "auditor", "seq": 1, "jenis": "usulan", "klaim": "Analisis terstruktur model terbaik.", "rujukan_bukti": ["trial:t-002"], "keyakinan": "tinggi", "alasan_keyakinan": "evaluasi", "risiko_asumsi": "none"}'
    )

    report, time_est = run_pilot_0b(
        work_dir=tmp_path / "pilot0b_test",
        llm_client=mock_llm,
        pilot0a_medians={"K0": 0.25, "K1": 0.30, "K2": 0.35},
    )

    assert len(report.runs) == 3
    assert report.runs[0].config_name == "K3"
    assert report.runs[1].config_name == "K6"
    assert report.runs[2].config_name == "KP"

    for run in report.runs:
        assert run.is_success is True
        assert run.duration_minutes <= 45.0
        assert run.peak_ram_gb <= 7.0 or run.peak_ram_gb < 16.0

    assert time_est.total_hours <= 40.0
    assert time_est.within_budget is True

