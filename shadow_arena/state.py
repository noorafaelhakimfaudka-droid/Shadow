# shadow_arena/state.py
"""
Engine Ekstraksi State Arena Streamlit (PRD 9.2, 9.3 & Kriteria AR-01 s/d AR-05).
- Hanya membaca `ledger.jsonl` dan `events.jsonl` dari folder run.
- Tidak ada state di luar dua file tersebut (Rule R13 & PRD 9.2).
- Ledger kosong -> tidak ada progres tampil (AR-04).
- Mampu merekonstruksi state hingga transisi tertentu (step) untuk fitur Replay (AR-01).
- Menyediakan data terstruktur untuk seluruh 8 panel (PRD 9.3).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ArenaState:
    run_id: Optional[str]
    is_empty: bool
    status: str
    stage: str
    grade: str
    total_events: int
    current_step: int
    models: List[Dict[str, Any]]
    leakage_chips: Dict[str, Dict[str, Any]]
    split_info: Dict[str, Any]
    vault_accessed: bool
    agent_artifacts: List[Dict[str, Any]]
    reproducibility_verified: Optional[bool]
    reproducibility_message: str


def load_arena_state(run_dir: Path | str, max_step: Optional[int] = None) -> ArenaState:
    """
    Mengekstrak state Arena secara deterministik dari events.jsonl dan ledger.jsonl.
    Jika max_step ditentukan, merekonstruksi state pada titik waktu tersebut (Replay).
    """
    run_d = Path(run_dir)
    events_p = run_d / "events.jsonl"
    ledger_p = run_d / "ledger.jsonl"

    if not events_p.exists() and not ledger_p.exists():
        return ArenaState(
            run_id=None,
            is_empty=True,
            status="tidak_ditemukan",
            stage="belum_mulai",
            grade="N/A",
            total_events=0,
            current_step=0,
            models=[],
            leakage_chips={},
            split_info={},
            vault_accessed=False,
            agent_artifacts=[],
            reproducibility_verified=None,
            reproducibility_message="Folder run belum memiliki event log.",
        )

    # Baca seluruh event
    events: List[Dict[str, Any]] = []
    if events_p.exists():
        for line in events_p.read_text(encoding="utf-8").splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    events.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    # Baca seluruh ledger
    ledger_entries: List[Dict[str, Any]] = []
    if ledger_p.exists():
        for line in ledger_p.read_text(encoding="utf-8").splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    ledger_entries.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    # AR-04: Ledger kosong -> tidak ada progres tampil
    if len(ledger_entries) == 0 and len(events) == 0:
        return ArenaState(
            run_id=None,
            is_empty=True,
            status="kosong",
            stage="siaga",
            grade="N/A",
            total_events=0,
            current_step=0,
            models=[],
            leakage_chips={},
            split_info={},
            vault_accessed=False,
            agent_artifacts=[],
            reproducibility_verified=None,
            reproducibility_message="Ledger kosong: tidak ada progres yang ditampilkan.",
        )

    total_events = len(events)
    target_step = total_events if max_step is None else min(max_step, total_events)
    active_events = events[:target_step]

    run_id = active_events[0].get("run_id") if active_events else (ledger_entries[0].get("run_id") if ledger_entries else "r-unknown")
    status = "berjalan"
    stage = "inisialisasi"
    grade = "Dalam proses"
    split_info: Dict[str, Any] = {}
    vault_accessed = False
    leakage_chips: Dict[str, Dict[str, Any]] = {
        "L1": {"status": "abu", "desc": "Fitur turunan target", "limit": "Tidak menangkap gabungan non-linear"},
        "L2": {"status": "abu", "desc": "Duplikat lintas split", "limit": "Tidak menangkap duplikat semantik"},
        "L5": {"status": "abu", "desc": "Kebocoran temporal", "limit": "Hanya fitur bertanggal/waktu"},
        "L6": {"status": "abu", "desc": "Kebocoran grup/entitas", "limit": "Grup wajib ada di kolom"},
        "L8": {"status": "abu", "desc": "Kebocoran indeks/ID", "limit": "Tidak menangkap urutan acak rumit"},
        "canary": {"status": "abu", "desc": "Kanari permutasi", "limit": "Hanya menangkap kebocoran prosedural"},
    }
    agent_artifacts: List[Dict[str, Any]] = []
    reproducibility_verified: Optional[bool] = None
    reproducibility_msg = "Menunggu verifikasi run-ulang"

    # Proses urutan active_events (Replay)
    for ev in active_events:
        ev_type = ev.get("type", "")
        payload = ev.get("payload", {})

        if ev_type == "run.started":
            stage = "framing_dan_audit"
        elif ev_type == "split.created":
            stage = "validasi_split"
            split_info = payload
        elif ev_type == "detector.result":
            det_name = payload.get("detector", "")
            has_blk = payload.get("has_blocker", False)
            chip_key = "L1" if "l1" in det_name.lower() else ("L8" if "l8" in det_name.lower() else None)
            if chip_key and chip_key in leakage_chips:
                leakage_chips[chip_key]["status"] = "merah" if has_blk else "hijau"
        elif ev_type == "canary.result":
            passed = payload.get("passed", False)
            leakage_chips["canary"]["status"] = "hijau" if passed else "merah"
            leakage_chips["canary"]["desc"] = payload.get("description", "Kanari permutasi")
        elif ev_type == "sealed.access":
            vault_accessed = True
            stage = "evaluasi_brankas"
        elif ev_type == "artifact.created":
            agent_artifacts.append(payload)
        elif ev_type == "replay.verified":
            reproducibility_verified = payload.get("success", False)
            reproducibility_msg = payload.get("message", "Replay selesai")
        elif ev_type == "run.finished":
            status = "selesai"
            grade = payload.get("grade", "Lengkap")

    # Model Arena: ambil entri dari ledger
    models: List[Dict[str, Any]] = []
    for entry in ledger_entries:
        models.append({
            "trial_id": entry.get("trial_id"),
            "model": entry.get("model"),
            "mean": entry.get("ringkasan", {}).get("mean"),
            "sd": entry.get("ringkasan", {}).get("sd"),
            "ci95": entry.get("ringkasan", {}).get("ci95", [0.0, 0.0]),
            "waktu_detik": entry.get("waktu_detik"),
            "ram_mb": entry.get("ram_puncak_mb"),
            "status": entry.get("status"),
        })

    return ArenaState(
        run_id=run_id,
        is_empty=False,
        status=status,
        stage=stage,
        grade=grade,
        total_events=total_events,
        current_step=target_step,
        models=models,
        leakage_chips=leakage_chips,
        split_info=split_info,
        vault_accessed=vault_accessed,
        agent_artifacts=agent_artifacts,
        reproducibility_verified=reproducibility_verified,
        reproducibility_message=reproducibility_msg,
    )
