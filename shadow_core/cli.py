# shadow_core/cli.py
"""
SHADOW Command Line Interface (CLI) (PRD 5.4).
Perintah:
- shadow run <csv> --target <col> --goal <desc> --mode <core|agents>
- shadow verify <run_dir>
- shadow purge <run_dir>
"""

from __future__ import annotations
import argparse
import sys
import shutil
from pathlib import Path

from shadow_core.runner import run_case, replay_and_verify_notebook
from shadow_core.manifest import Manifest
from shadow_core.binder import verify, extract_ledger_numbers
from shadow_core.ledger import Ledger


def cmd_run(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: File CSV '{csv_path}' tidak ditemukan!")
        return 1

    print(f"Memulai SHADOW run pada {csv_path} (target={args.target}, mode={args.mode})...")
    res = run_case(
        csv_path=csv_path,
        target_col=args.target,
        goal=args.goal or "Analisis ML Tervalidasi",
        mode=args.mode,
        run_dir=args.out_dir,
    )

    print("\n" + "=" * 60)
    print(f"SHADOW Run Selesai: {res.run_id}")
    print(f"Nilai Penyerahan : {res.grade}")
    print(f"Model Terpilih   : {res.best_model_name}")
    print(f"Waktu Eksekusi   : {res.execution_time_seconds:.1f}s")
    print(f"Lokasi Notebook  : {res.notebook_path}")
    print("=" * 60)
    return 0 if res.is_success else 1


def cmd_verify(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Direktori run '{run_dir}' tidak ditemukan!")
        return 1

    nb_path = run_dir / "notebook.ipynb"
    if not nb_path.exists():
        print(f"Error: Notebook tidak ditemukan di {run_dir}!")
        return 1

    print(f"Memverifikasi ulang secara bersih: {nb_path}...")
    success, msg = replay_and_verify_notebook(nb_path, run_dir=run_dir)
    print(f"Hasil Replay: {'SUKSES' if success else 'GAGAL'} - {msg}")

    # Cek binder
    ledger_path = run_dir / "ledger.jsonl"
    if ledger_path.exists():
        ledger = Ledger(ledger_path)
        nb_text = nb_path.read_text(encoding="utf-8")
        violations = verify(nb_text, ledger)
        print(f"Pemeriksaan Pengikat Angka (Binder): {len(violations)} pelanggaran.")
        for v in violations[:5]:
            print(f"  - [{v.token}]: {v.message}")

    return 0 if success else 1


def cmd_purge(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run '{run_dir}' tidak ditemukan!")
        return 1

    print(f"Menghapus seluruh artefak kasus di {run_dir}...")
    shutil.rmtree(run_dir)
    print("Penghapusan selesai.")
    return 0


def cmd_arena(args: argparse.Namespace) -> int:
    import subprocess
    app_path = Path(__file__).resolve().parent.parent / "shadow_arena" / "app.py"
    print(f"Membuka SHADOW Arena (Streamlit) pada {app_path}...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
        return 0
    except (KeyboardInterrupt, subprocess.SubprocessError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="shadow", description="SHADOW: Laboratorium Eksperimen ML Lokal & Auditable")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # shadow run
    run_parser = subparsers.add_parser("run", help="Jalankan alur eksperimen ML")
    run_parser.add_argument("csv", help="Jalur file CSV input")
    run_parser.add_argument("--target", required=True, help="Nama kolom target")
    run_parser.add_argument("--goal", default="", help="Tujuan satu kalimat")
    run_parser.add_argument("--mode", default="core", choices=["core", "agents"], help="Mode eksekusi")
    run_parser.add_argument("--out-dir", default=None, help="Direktori keluaran run")
    run_parser.set_defaults(func=cmd_run)

    # shadow verify
    verify_parser = subparsers.add_parser("verify", help="Verifikasi ulang bersih notebook run")
    verify_parser.add_argument("run_dir", help="Jalur folder run")
    verify_parser.set_defaults(func=cmd_verify)

    # shadow purge
    purge_parser = subparsers.add_parser("purge", help="Hapus seluruh artefak kasus run")
    purge_parser.add_argument("run_dir", help="Jalur folder run")
    purge_parser.set_defaults(func=cmd_purge)

    # shadow arena
    arena_parser = subparsers.add_parser("arena", help="Buka halaman Streamlit Arena (M2)")
    arena_parser.set_defaults(func=cmd_arena)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
