# tests/test_ledger_manifest.py
import pytest
from pathlib import Path
from shadow_core.ledger import Ledger, LedgerError
from shadow_core.events import EventLog
from shadow_core.manifest import Manifest, hash_file


def test_ledger_append_only(tmp_path: Path):
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_file)

    entry1 = {
        "trial_id": "t-001",
        "run_id": "run-01",
        "case_id": "case-01",
        "ts": "2026-10-12T10:00:00Z",
        "agent": "experiment_engineer",
        "hipotesis": "Baseline model test",
        "model": "LogisticRegression",
        "hiperparameter": {"C": 1.0},
        "pipeline_hash": "sha256:111",
        "fold_def_hash": "sha256:222",
        "seed": 42,
        "skor_per_lipatan": [0.8, 0.82, 0.79],
        "ringkasan": {"mean": 0.803, "sd": 0.015, "ci95": [0.78, 0.82]},
        "waktu_detik": 2.5,
        "ram_puncak_mb": 120,
        "status": "selesai",
        "alasan": None,
        "cell_notebook": "7.1",
        "akses_uji_tersegel": False,
    }

    t_id = ledger.append(entry1)
    assert t_id == "t-001"
    assert len(ledger) == 1

    # Percobaan duplikasi ID harus ditolak keras
    with pytest.raises(LedgerError):
        ledger.append(entry1)

    # Entri tanpa atribut wajib harus ditolak
    with pytest.raises(LedgerError):
        ledger.append({"trial_id": "t-002", "run_id": "run-01"})


def test_event_log(tmp_path: Path):
    event_file = tmp_path / "events.jsonl"
    events = EventLog(event_file, run_id="run-01")

    ev1 = events.log("run.started", {"mode": "core", "input": "test.csv"})
    assert ev1["seq"] == 1
    assert ev1["type"] == "run.started"

    ev2 = events.log("split.created", {"folds": 5})
    assert ev2["seq"] == 2

    all_ev = events.read()
    assert len(all_ev) == 2
    assert events.get_by_type("run.started")[0]["seq"] == 1


def test_manifest_source_hash(tmp_path: Path):
    source_file = tmp_path / "data.csv"
    source_file.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest = Manifest(manifest_file)

    split_info = {"strategy": "StratifiedKFold", "seed": 42}
    data = manifest.create(
        run_id="run-01",
        mode="core",
        input_file=source_file,
        split_info=split_info,
    )

    initial_hash = data["input"]["sha256"]
    assert len(initial_hash) == 64

    # File tidak berubah
    assert manifest.finalize_source_check(source_file) is True

    # Jika file berubah, deteksi ketidakcocokan
    source_file.write_text("a,b,c\n1,2,3\n4,5,999\n", encoding="utf-8")
    assert manifest.finalize_source_check(source_file) is False
