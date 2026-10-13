# tests/test_benchmark_m3.py
"""
Uji Unit untuk Harness Benchmark M3 & Evaluasi Akhir (PRD 11.3, 11.4, 12.4).
- 30 Kasus Simpanan Held-Out (>= 30% kontrol bersih)
- Eksekusi alur benchmark dan adjudikasi dua penilai
- Keputusan pilih_default tervalidasi bila N >= 30 dan protokol_penuh = True
"""

from pathlib import Path
import pytest
from shadow_agents.llm_local import LocalLLMClient
from shadow_eval.benchmark_m3 import (
    prepare_m3_held_out_cases,
    run_m3_benchmark,
    M3BenchmarkReport,
)


def test_prepare_m3_held_out_cases(tmp_path: Path):
    """Memverifikasi penyiapan 30 kasus simpanan held-out dengan >= 30% kontrol bersih."""
    cases = prepare_m3_held_out_cases(tmp_path, n_cases=30)
    assert len(cases) == 30

    clean_cases = [c for c in cases if len(c.catalog) == 0]
    # PRD 10.2: Kasus kontrol bersih >= 30%
    clean_pct = len(clean_cases) / len(cases)
    assert clean_pct >= 0.30

    # Pastikan setiap file ada
    for c in cases:
        assert c.csv_path.exists()
        assert c.defects_yaml_path.exists()


def test_run_m3_benchmark_fast_mode(tmp_path: Path):
    """
    Menjalankan pengujian integrasi alur benchmark M3 dalam fast_mode.
    Memverifikasi evaluasi K0..KP, adjudikasi dua penilai, dan keputusan akhir.
    """
    mock_llm = LocalLLMClient(
        mock_response='{"artifact_id": "a-m3-01", "run_id": "r-01", "agent": "auditor", "seq": 1, "jenis": "temuan", "klaim": "Temuan audit.", "rujukan_bukti": ["trial:t-01"], "keyakinan": "tinggi", "alasan_keyakinan": "audit", "risiko_asumsi": "none"}'
    )

    all_cases = prepare_m3_held_out_cases(tmp_path / "m3_prep", n_cases=5)

    report = run_m3_benchmark(
        work_dir=tmp_path / "m3_run",
        cases=all_cases,
        llm_client=mock_llm,
        fast_mode=True,
    )

    # 5 kasus x 6 configs x 1 rep = 30 runs
    assert report.total_cases == 5
    assert report.total_runs == 30
    assert report.completed_runs == 30
    assert len(report.metrics_by_config) == 6
    assert report.agreement_report.overall_agreement_pct >= 0.85
    assert report.decision.default_config in ["K0", "K1", "K2", "K3", "K6"]
    # 5 kasus < 30 -> status 'indikatif'
    assert report.decision.status == "indikatif"
