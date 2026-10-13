# tests/test_cli.py
"""
Uji Fungsionalitas CLI SHADOW (PRD 5.4, FR-22 & AC-18):
- shadow run
- shadow verify
- shadow purge
"""

import argparse
from pathlib import Path
import pytest
import pandas as pd
from shadow_core.cli import cmd_run, cmd_verify, cmd_purge
from shadow_eval.seed_defects import generate_clean_dataset


def test_cli_run_verify_purge(tmp_path):
    """Menguji alur lengkap perintah CLI: run -> verify -> purge."""
    csv_file = tmp_path / "cli_test_data.csv"
    df = generate_clean_dataset(n_samples=250, seed=42)
    df.to_csv(csv_file, index=False)

    run_dir = tmp_path / "cli_run_out"

    # 1. Test 'shadow run'
    args_run = argparse.Namespace(
        csv=str(csv_file),
        target="target",
        goal="Test CLI Run",
        mode="core",
        out_dir=str(run_dir),
    )
    ret_run = cmd_run(args_run)
    assert ret_run == 0, f"CLI run gagal dengan return code {ret_run}"
    assert (run_dir / "ledger.jsonl").exists()
    assert (run_dir / "notebook.ipynb").exists()
    assert (run_dir / "manifest.json").exists()

    # 2. Test 'shadow verify'
    args_verify = argparse.Namespace(run_dir=str(run_dir))
    ret_verify = cmd_verify(args_verify)
    assert ret_verify == 0, f"CLI verify gagal dengan return code {ret_verify}"

    # 3. Test 'shadow purge'
    args_purge = argparse.Namespace(run_dir=str(run_dir))
    ret_purge = cmd_purge(args_purge)
    assert ret_purge == 0, f"CLI purge gagal dengan return code {ret_purge}"
    assert not run_dir.exists(), "Direktori run masih ada setelah shadow purge!"


def test_cli_arena_parser():
    """Menguji bahwa subparser 'arena' terdaftar dengan benar di CLI."""
    from shadow_core.cli import main
    import sys
    # Pastikan file app.py arena ada
    arena_app = Path(__file__).resolve().parent.parent / "shadow_arena" / "app.py"
    assert arena_app.exists(), "shadow_arena/app.py tidak ditemukan!"
