# tests/test_single_agent_k1_k2.py
"""
Uji Single Agent K1, K2, dan Jaring Pengaman Core (PRD 12.2 T1.4 & AC-16).
- K1: 1 agen menyusun narasi dan klaim terstruktur
- K2: 1 agen + tinjau diri (self-critique)
- AC-16: Layanan model dimatikan di tengah run -> status tercatat jelas,
  Core melanjutkan secara deterministik tanpa membuat notebook/klaim palsu.
"""

import pytest
from pathlib import Path
from shadow_core.ledger import Ledger
from shadow_core.events import EventLog
from shadow_agents.llm_local import LocalLLMClient, LLMConfig
from shadow_agents.single_agent import SingleAgentOrchestrator


@pytest.fixture
def run_env(tmp_path):
    events_p = tmp_path / "events.jsonl"
    ledger_p = tmp_path / "ledger.jsonl"
    event_log = EventLog(events_p, run_id="r-test-01")
    ledger = Ledger(ledger_p)

    # Tambahkan percobaan acuan di ledger
    ledger.append({
        "trial_id": "t-002", "run_id": "r-test-01", "case_id": "c-01",
        "agent": "core", "hipotesis": "Linear model baseline",
        "model": "LogisticRegression", "hiperparameter": {},
        "pipeline_hash": "h1", "fold_def_hash": "f1", "seed": 42,
        "skor_per_lipatan": [0.80, 0.82, 0.81],
        "ringkasan": {"mean": 0.8100, "sd": 0.01},
        "waktu_detik": 2.0, "ram_puncak_mb": 150,
        "status": "selesai", "alasan": None, "cell_notebook": "8.2",
        "akses_uji_tersegel": False,
    })

    return event_log, ledger


def test_k1_single_agent_flow(run_env):
    """Menguji eksekusi K1: 1 agen menyusun klaim terstruktur."""
    import json
    event_log, ledger = run_env
    valid_json_resp = json.dumps({
        "artifact_id": "a-k1-01",
        "run_id": "r-test-01",
        "agent": "k1_assistant",
        "seq": 1,
        "jenis": "temuan",
        "klaim": "Model terpilih menunjukkan performa konsisten pada validasi silang.",
        "rujukan_bukti": ["trial:t-002"],
        "keyakinan": "tinggi",
        "alasan_keyakinan": "evaluasi Core",
        "risiko_asumsi": "asumsi normal",
        "structured_claim": {
            "subject": "LogisticRegression",
            "metric": "roc_auc",
            "split": "cv",
            "trial_key": "t-002",
            "template": "Model {{subjek}} memperoleh performa {{nilai}}.",
        }
    })
    client = LocalLLMClient(LLMConfig(host="127.0.0.1:11434"), mock_response=valid_json_resp)
    orch = SingleAgentOrchestrator(llm_client=client, event_log=event_log)

    res = orch.run_k1(
        run_id="r-test-01",
        case_id="c-01",
        ledger=ledger,
        target_col="target",
        best_model_name="LogisticRegression",
    )

    assert res.mode == "K1"
    assert len(res.artifacts) >= 1
    art = res.artifacts[0]
    assert art.structured_claim is not None
    assert art.structured_claim.subject == "LogisticRegression"
    assert any("t-002" in r for r in art.rujukan_bukti)


def test_k2_self_critique_flow(run_env):
    """Menguji eksekusi K2: 1 agen + tinjau diri (self-critique)."""
    import json
    event_log, ledger = run_env
    valid_json_resp = json.dumps({
        "artifact_id": "a-k1-01",
        "run_id": "r-test-01",
        "agent": "k1_assistant",
        "seq": 1,
        "jenis": "temuan",
        "klaim": "Model terpilih menunjukkan performa konsisten pada validasi silang.",
        "rujukan_bukti": ["trial:t-002"],
        "keyakinan": "tinggi",
        "alasan_keyakinan": "evaluasi Core",
        "risiko_asumsi": "asumsi normal",
        "structured_claim": {
            "subject": "LogisticRegression",
            "metric": "roc_auc",
            "split": "cv",
            "trial_key": "t-002",
            "template": "Model {{subjek}} memperoleh performa {{nilai}}.",
        }
    })
    client = LocalLLMClient(LLMConfig(host="127.0.0.1:11434"), mock_response=valid_json_resp)
    orch = SingleAgentOrchestrator(llm_client=client, event_log=event_log)


    res = orch.run_k2(
        run_id="r-test-01",
        case_id="c-01",
        ledger=ledger,
        target_col="target",
        best_model_name="LogisticRegression",
    )

    assert res.mode == "K2"
    assert len(res.artifacts) >= 2
    # Ada artefak awal dan artefak tinjau diri
    critique_artifacts = [a for a in res.artifacts if a.agent == "k2_self_reviewer"]
    assert len(critique_artifacts) >= 1
    assert critique_artifacts[0].jenis == "kritik"
    assert res.self_critique_passed is True


def test_ac16_llm_service_failure_safety_net(run_env):
    """AC-16: Simulasi layanan LLM mati/port tak aktif -> Core lanjut secara aman tanpa crash."""
    event_log, ledger = run_env
    # Hubungkan ke port lokal yang sengaja mati/tidak ada listener
    dead_client = LocalLLMClient(LLMConfig(host="127.0.0.1:59999", timeout_seconds=1))
    orch = SingleAgentOrchestrator(llm_client=dead_client, event_log=event_log)

    res = orch.run_k1(
        run_id="r-test-01",
        case_id="c-01",
        ledger=ledger,
        target_col="target",
        best_model_name="LogisticRegression",
    )

    # Core safety net berhasil menangani: status tercatat, fallback aman digunakan
    assert res.llm_available is False
    assert res.fallback_used is True
    assert len(res.artifacts) >= 1
    assert res.artifacts[0].structured_claim.subject == "LogisticRegression"

    # Periksa bahwa event failure tercatat di events.jsonl
    failure_events = event_log.get_by_type("agent.failure")
    assert len(failure_events) == 1
    assert "Layanan LLM lokal mati" in failure_events[0]["payload"]["reason"]
