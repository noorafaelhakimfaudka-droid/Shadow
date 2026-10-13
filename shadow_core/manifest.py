# shadow_core/manifest.py
"""
Manifest: Menyimpan metadata eksekusi, hash reproduksibilitas, dependensi, dan status penyerahan (PRD Lampiran B.2).
Aturan R9: Sumber read-only; hash diperiksa sebelum dan sesudah run.
"""

from __future__ import annotations
import hashlib
import json
import platform
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def hash_file(path: Path | str) -> str:
    """Menghitung hash SHA-256 dari sebuah file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan untuk hashing: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_hardware_info() -> Dict[str, Any]:
    """Mendeteksi spesifikasi hardware laptop lokal."""
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    cpu_info = platform.processor() or platform.machine()
    return {
        "ram_gb": ram_gb,
        "cpu": cpu_info,
        "os": platform.system(),
        "os_version": platform.version(),
    }


class Manifest:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            self.load()

    def create(
        self,
        run_id: str,
        mode: str,
        input_file: Path | str,
        split_info: Dict[str, Any],
        dependencies_file: Optional[Path | str] = None,
        llm_info: Optional[Dict[str, Any]] = None,
        toleransi_ulang: float = 0.001,
    ) -> Dict[str, Any]:
        """Inisialisasi manifest baru."""
        inp_path = Path(input_file)
        dep_path = Path(dependencies_file) if dependencies_file else None

        self.data = {
            "shadow_version": "4.1",
            "run_id": run_id,
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "file": str(inp_path.name),
                "path": str(inp_path.resolve()),
                "sha256": hash_file(inp_path),
                "sha256_sesudah": None,
            },
            "split": split_info,
            "dependensi": str(dep_path.name) if dep_path else "requirements.lock",
            "dependensi_sha256": hash_file(dep_path) if dep_path and dep_path.exists() else None,
            "llm": llm_info or {
                "runtime": "none",
                "model_tag": None,
                "kuantisasi": "none",
                "suhu": 0,
                "seed": split_info.get("seed", 42),
            },
            "toleransi_ulang": toleransi_ulang,
            "perangkat": get_hardware_info(),
            "waktu_total_menit": 0.0,
            "status_cell": {},
            "penyerahan": "Dalam Pengerjaan",
        }
        self.save()
        return self.data

    def finalize_source_check(self, input_file: Path | str) -> bool:
        """Memeriksa hash file sumber sesudah eksekusi (Aturan R9)."""
        inp_path = Path(input_file)
        hash_after = hash_file(inp_path)
        self.data["input"]["sha256_sesudah"] = hash_after
        self.save()
        return self.data["input"]["sha256"] == hash_after

    def update_grade(self, grade: str) -> None:
        """Memperbarui nilai penyerahan: Lengkap, Dengan peringatan, Parsial, Ditolak."""
        self.data["penyerahan"] = grade
        self.save()

    def update_waktu(self, menit: float) -> None:
        self.data["waktu_total_menit"] = round(menit, 2)
        self.save()

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
