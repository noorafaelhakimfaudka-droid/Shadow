# shadow_core/ledger.py
"""
Ledger: Catatan append-only (JSONL) semua percobaan eksekusi model (PRD 5.3 & Lampiran B.1).
Aturan R5: Semua percobaan masuk ledger, termasuk yang gagal/jelek.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_FIELDS = {
    "trial_id",
    "run_id",
    "case_id",
    "ts",
    "agent",
    "hipotesis",
    "model",
    "hiperparameter",
    "pipeline_hash",
    "fold_def_hash",
    "seed",
    "skor_per_lipatan",
    "ringkasan",
    "waktu_detik",
    "ram_puncak_mb",
    "status",
    "alasan",
    "cell_notebook",
    "akses_uji_tersegel",
}


class LedgerError(Exception):
    """Kesalahan pada operasi ledger."""


class Ledger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, entry: Dict[str, Any]) -> str:
        """Menambahkan satu baris percobaan ke ledger.
        Tidak ada operasi update atau delete."""
        current_entries = self.read()
        existing_ids = {e["trial_id"] for e in current_entries}

        if "trial_id" not in entry or not entry["trial_id"]:
            next_num = len(current_entries) + 1
            entry["trial_id"] = f"t-{next_num:03d}"

        trial_id = entry["trial_id"]
        if trial_id in existing_ids:
            raise LedgerError(f"Pelanggaran append-only: trial_id '{trial_id}' sudah ada!")

        if "ts" not in entry or not entry["ts"]:
            entry["ts"] = datetime.now(timezone.utc).isoformat()

        # Validasi field wajib
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise LedgerError(f"Entri ledger tidak lengkap, atribut kurang: {missing}")

        # Pastikan ringkasan memiliki mean, sd, ci95
        ringkasan = entry.get("ringkasan", {})
        if not isinstance(ringkasan, dict) or "mean" not in ringkasan:
            raise LedgerError("Entri ringkasan wajib berupa dictionary dan memiliki atribut 'mean'.")

        serialized = json.dumps(entry, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(serialized + "\n")
            f.flush()

        return trial_id

    def read(self) -> List[Dict[str, Any]]:
        """Membaca seluruh entri dari ledger.jsonl."""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    entries.append(json.loads(clean))
                except json.JSONDecodeError as err:
                    raise LedgerError(f"Ledger rusak pada baris {line_no}: {err}") from err
        return entries

    def get(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Mengambil satu entri berdasarkan trial_id."""
        for entry in self.read():
            if entry.get("trial_id") == trial_id:
                return entry
        return None

    def __len__(self) -> int:
        return len(self.read())
