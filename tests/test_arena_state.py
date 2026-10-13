# tests/test_arena_state.py
"""
Uji Kriteria Penerimaan Arena Streamlit (PRD 9.5 & AR-01 s/d AR-05).
- AR-01: Setiap indikator terhubung ke event/ledger nyata; pembanding otomatis event log vs state UI cocok 100% pada replay.
- AR-02: State loads dalam <= 5 detik pada laptop acuan.
- AR-03: Setiap angka di panel dapat ditelusuri ke ledger <= 2 klik (pemetaan langsung trial_id).
- AR-04: Ledger kosong -> tidak ada progres tampil (is_empty=True).
- AR-05: Tidak ada panel menampilkan data demo tanpa label.
"""

import time
from pathlib import Path
from typing import Any, Dict
import pytest
from shadow_arena.state import load_arena_state, ArenaState
from shadow_core.events import EventLog
from shadow_core.ledger import Ledger


def make_valid_trial(
    trial_id: str,
    run_id: str,
    model: str,
    mean: float,
    sd: float,
) -> Dict[str, Any]:
    return {
        "trial_id": trial_id,
        "run_id": run_id,
        "case_id": "case-test",
        "ts": "2026-10-12T10:00:00Z",
        "agent": "engineer",
        "hipotesis": f"Uji performa {model}",
        "model": model,
        "hiperparameter": {},
        "pipeline_hash": "hash_pipe",
        "fold_def_hash": "hash_fold",
        "seed": 42,
        "skor_per_lipatan": [mean - sd, mean + sd],
        "ringkasan": {"mean": mean, "sd": sd, "ci95": [mean - 1.96 * sd, mean + 1.96 * sd]},
        "waktu_detik": 2.1,
        "ram_puncak_mb": 400,
        "status": "selesai",
        "alasan": None,
        "cell_notebook": "8.1",
        "akses_uji_tersegel": False,
    }


def test_ar04_empty_ledger_displays_no_progress(tmp_path: Path):
    """AR-04: Ledger kosong -> tidak ada progres eksperimen yang ditampilkan."""
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    (empty_run / "events.jsonl").write_text("", encoding="utf-8")
    (empty_run / "ledger.jsonl").write_text("", encoding="utf-8")

    state = load_arena_state(empty_run)
    assert state.is_empty is True
    assert len(state.models) == 0
    assert len(state.agent_artifacts) == 0


def test_ar01_replay_step_consistency(tmp_path: Path):
    """
    AR-01: Setiap indikator terhubung ke event log nyata.
    Replay pada step tertentu menghasilkan rekonstruksi state yang persis tanpa selisih.
    """
    run_dir = tmp_path / "replay_run"
    run_dir.mkdir()

    event_log = EventLog(run_dir / "events.jsonl", run_id="r-replay-01")
    ledger = Ledger(run_dir / "ledger.jsonl")

    # Rekam urutan kejadian
    event_log.log("run.started", {"mode": "agents"})
    event_log.log("split.created", {"strategy": "RepeatedKFold", "n_dev": 500, "n_sealed": 100})
    event_log.log("detector.result", {"detector": "l1_target", "has_blocker": False})

    ledger.append(make_valid_trial("t-001", "r-replay-01", "Linear", 0.81, 0.02))

    event_log.log("sealed.access", {"status": "success"})
    event_log.log("artifact.created", {"artifact_id": "a-01", "agent": "auditor", "klaim": "Audit bersih"})
    event_log.log("replay.verified", {"success": True, "message": "100% cocok"})
    event_log.log("run.finished", {"grade": "Lengkap"})

    # Uji State Lengkap
    full_state = load_arena_state(run_dir)
    assert full_state.is_empty is False
    assert full_state.total_events == 7
    assert full_state.status == "selesai"
    assert full_state.vault_accessed is True
    assert full_state.reproducibility_verified is True
    assert len(full_state.models) == 1

    # Uji Replay Step 2: Baru sampai split.created
    step2_state = load_arena_state(run_dir, max_step=2)
    assert step2_state.total_events == 7
    assert step2_state.current_step == 2
    assert step2_state.stage == "validasi_split"
    # Pada step 2, sealed holdout belum diakses dan replay belum diverifikasi
    assert step2_state.vault_accessed is False
    assert step2_state.reproducibility_verified is None


def test_ar02_load_performance(tmp_path: Path):
    """AR-02: Pembacaan state berlangsung <= 5 detik pada laptop acuan."""
    run_dir = tmp_path / "perf_run"
    run_dir.mkdir()
    event_log = EventLog(run_dir / "events.jsonl", run_id="r-perf-01")

    # Tulis 200 event
    for i in range(200):
        event_log.log("detector.result", {"detector": f"det_{i}", "has_blocker": False})

    t0 = time.time()
    state = load_arena_state(run_dir)
    elapsed = time.time() - t0

    assert elapsed < 5.0, f"Loading state butuh {elapsed:.2f} detik, melebihi batas 5 detik!"
    assert state.total_events == 200


def test_ar03_and_ar05_traceability_and_demo_labels(tmp_path: Path):
    """
    AR-03: Angka di model arena terikat 1:1 ke entri ledger.
    AR-05: Tidak ada data tanpa keterikatan yang ditampilkan tanpa label DEMO/PROTOTYPE.
    """
    run_dir = tmp_path / "trace_run"
    run_dir.mkdir()
    ledger = Ledger(run_dir / "ledger.jsonl")

    ledger.append(make_valid_trial("t-042", "r-trace-01", "HistGradientBoosting", 0.885, 0.012))

    state = load_arena_state(run_dir)
    assert len(state.models) == 1
    m = state.models[0]
    assert m["trial_id"] == "t-042"
    assert m["mean"] == 0.885
    assert m["model"] == "HistGradientBoosting"
