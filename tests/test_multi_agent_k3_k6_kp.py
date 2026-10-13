# tests/test_multi_agent_k3_k6_kp.py
"""
Uji Orkestrasi Multi-Agen K3, K6, dan KP (Plasebo) (PRD 8.3, 11.1, 12.3 T2.1 & T2.2).
"""

from pathlib import Path
import pytest
from shadow_agents.orchestrator import MultiAgentOrchestrator, IRRELEVANT_PLACEBO_CRITIQUES
from shadow_agents.llm_local import LocalLLMClient
from shadow_core.events import EventLog
from shadow_core.ledger import Ledger


@pytest.fixture
def dummy_ledger(tmp_path: Path) -> Ledger:
    ledger_p = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_p)
    ledger.append({
        "trial_id": "t-001",
        "run_id": "r-001",
        "case_id": "case-01",
        "ts": "2026-10-12T10:00:00Z",
        "agent": "engineer",
        "hipotesis": "Baseline dummy",
        "model": "Dummy",
        "hiperparameter": {},
        "pipeline_hash": "hash_123",
        "fold_def_hash": "fold_123",
        "seed": 42,
        "skor_per_lipatan": [0.5, 0.5],
        "ringkasan": {"mean": 0.5, "sd": 0.0},
        "waktu_detik": 1.0,
        "ram_puncak_mb": 500,
        "status": "selesai",
        "alasan": None,
        "cell_notebook": "8.1",
        "akses_uji_tersegel": False,
    })
    ledger.append({
        "trial_id": "t-002",
        "run_id": "r-001",
        "case_id": "case-01",
        "ts": "2026-10-12T10:01:00Z",
        "agent": "engineer",
        "hipotesis": "HistGradientBoosting unggul",
        "model": "HistGradientBoosting",
        "hiperparameter": {},
        "pipeline_hash": "hash_456",
        "fold_def_hash": "fold_123",
        "seed": 42,
        "skor_per_lipatan": [0.82, 0.84],
        "ringkasan": {"mean": 0.83, "sd": 0.01},
        "waktu_detik": 2.0,
        "ram_puncak_mb": 600,
        "status": "selesai",
        "alasan": None,
        "cell_notebook": "8.2",
        "akses_uji_tersegel": False,
    })
    return ledger


@pytest.fixture
def mock_llm():
    return LocalLLMClient(
        mock_response='{"artifact_id": "a-mock", "run_id": "r-001", "agent": "auditor", "seq": 1, "jenis": "usulan", "klaim": "Analisis terstruktur tanpa angka bebas.", "rujukan_bukti": ["trial:t-002"], "keyakinan": "tinggi", "alasan_keyakinan": "evaluasi", "risiko_asumsi": "none"}'
    )


def test_k3_orchestration(tmp_path: Path, dummy_ledger: Ledger, mock_llm: LocalLLMClient):
    """Memverifikasi konfigurasi K3 menjalankan tepat 3 agen spesialis."""
    event_log = EventLog(tmp_path / "events.jsonl", run_id="r-k3-01")
    orchestrator = MultiAgentOrchestrator(event_log=event_log, llm_client=mock_llm)

    res = orchestrator.run_k3(
        run_id="r-k3-01",
        case_id="case-k3",
        ledger=dummy_ledger,
        audit_summary={"n_rows": 500, "n_cols": 10},
        detector_results={"L1": {"passed": True}, "L2": {"passed": True}},
        best_model_name="HistGradientBoosting",
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        delta=0.01,
    )

    assert res.is_success is True
    assert res.mode == "K3"
    assert len(res.artifacts) == 3

    agents_in_run = [a.agent for a in res.artifacts]
    assert agents_in_run == ["auditor", "engineer", "skeptic"]


def test_k6_orchestration_and_critique_rounds(tmp_path: Path, dummy_ledger: Ledger, mock_llm: LocalLLMClient):
    """Memverifikasi konfigurasi K6 menjalankan 6 agen berurutan + putaran kritik silang."""
    event_log = EventLog(tmp_path / "events.jsonl", run_id="r-k6-01")
    orchestrator = MultiAgentOrchestrator(event_log=event_log, llm_client=mock_llm, max_critique_rounds=3)

    res = orchestrator.run_k6(
        run_id="r-k6-01",
        case_id="case-k6",
        ledger=dummy_ledger,
        task_type="binary_classification",
        primary_metric="roc_auc",
        candidates=["Linear", "RandomForest", "HistGradientBoosting"],
        delta=0.01,
        audit_summary={"n_rows": 500, "n_cols": 10},
        detector_results={"L1": {"passed": True}},
        best_model_name="HistGradientBoosting",
        canary_passed=True,
        canary_description="Kanari permutasi lolos",
        seed_variance_cv=0.01,
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        is_placebo=False,
    )

    assert res.is_success is True
    assert res.mode == "K6"
    # 6 agen dasar + 2 putaran kritik tambahan = 8 artefak
    assert len(res.artifacts) == 8
    assert res.critique_rounds == 3

    base_agents = [a.agent for a in res.artifacts[:6]]
    assert base_agents == [
        "auditor",
        "architect",
        "feature_strategist",
        "engineer",
        "robustness",
        "skeptic",
    ]


def test_kp_placebo_orchestration(tmp_path: Path, dummy_ledger: Ledger, mock_llm: LocalLLMClient):
    """
    Memverifikasi konfigurasi KP (Plasebo):
    Memiliki struktur, urutan peran, dan putaran kritik yang sama dengan K6,
    tetapi isi kritik diacak dari kumpulan tidak relevan.
    """
    event_log = EventLog(tmp_path / "events.jsonl", run_id="r-kp-01")
    orchestrator = MultiAgentOrchestrator(event_log=event_log, llm_client=mock_llm, max_critique_rounds=3)

    res = orchestrator.run_k6(
        run_id="r-kp-01",
        case_id="case-kp",
        ledger=dummy_ledger,
        task_type="binary_classification",
        primary_metric="roc_auc",
        candidates=["Linear", "HistGradientBoosting"],
        delta=0.01,
        audit_summary={"n_rows": 500, "n_cols": 10},
        detector_results={"L1": {"passed": True}},
        best_model_name="HistGradientBoosting",
        canary_passed=True,
        canary_description="Kanari permutasi lolos",
        seed_variance_cv=0.01,
        comparison_summary={"ci95_lower": 0.05, "ci95_upper": 0.15},
        is_placebo=True,
    )

    assert res.is_success is True
    assert res.mode == "KP"
    assert len(res.artifacts) == 8

    # Periksa bahwa kritik pada KP berasal dari pool plasebo
    critiques = [a for a in res.artifacts if a.jenis == "kritik" and a.status == "belum_terselesaikan"]
    assert len(critiques) >= 2
    for c in critiques:
        assert c.klaim in IRRELEVANT_PLACEBO_CRITIQUES
