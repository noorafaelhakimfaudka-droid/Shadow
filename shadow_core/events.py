# shadow_core/events.py
"""
Event Log: Catatan kejadian terurut (events.jsonl) untuk auditabilitas dan Streamlit Arena (PRD 5.6).
Tipe wajib:
run.started, split.created, detector.result, canary.result, trial.completed,
sealed.access, artifact.created, approval.requested, claim.linked, replay.verified, run.finished.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MANDATORY_EVENT_TYPES = {
    "run.started",
    "split.created",
    "detector.result",
    "canary.result",
    "trial.completed",
    "sealed.access",
    "artifact.created",
    "approval.requested",
    "claim.linked",
    "replay.verified",
    "run.finished",
}


class EventLogError(Exception):
    """Kesalahan pada penulisan atau pembacaan event log."""


class EventLog:
    def __init__(self, path: Path | str, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Menuliskan satu event ke events.jsonl."""
        current_events = self.read()
        seq = len(current_events) + 1

        event = {
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "type": event_type,
            "payload": payload,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

        return event

    def read(self) -> List[Dict[str, Any]]:
        """Membaca seluruh event yang tercatat."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    events.append(json.loads(clean))
                except json.JSONDecodeError as err:
                    raise EventLogError(f"Event log rusak pada baris {line_no}: {err}") from err
        return events

    def get_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Mengambil semua event dengan tipe tertentu."""
        return [e for e in self.read() if e.get("type") == event_type]
