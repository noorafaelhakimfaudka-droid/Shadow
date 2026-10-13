# tests/test_v6_reproducibility.py
"""
Verifikasi V6: Reproducibility Run-Ulang Bersih (PRD 10.1 & 6.10).
Target [T]: 100% run menghasilkan notebook yang lolos re-eksekusi bersih dan metrik cocok <= 0.001.
"""

import pytest
from pathlib import Path
import json
import pandas as pd
from shadow_core.runner import run_case, replay_and_verify_notebook
from shadow_eval.seed_defects import generate_clean_dataset


def test_v6_end_to_end_reproducible_run(tmp_path: Path):
    csv_file = tmp_path / "data_clean.csv"
    df = generate_clean_dataset(n_samples=600, task="binary", seed=42)
    df.to_csv(csv_file, index=False)

    run_dir = tmp_path / "run_repro"
    res = run_case(
        csv_path=csv_file,
        target_col="target",
        goal="Prediksi klasifikasi biner tervalidasi",
        run_dir=run_dir,
    )

    print(f"\nV6 Run Result Grade: {res.grade}")
    print(f"V6 Execution Time  : {res.execution_time_seconds:.1f}s")
    print(f"V6 Best Model      : {res.best_model_name}")

    assert res.is_success is True
    assert res.notebook_path.exists()
    assert (run_dir / "ledger.jsonl").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "vault.spent").exists()

    # Re-eksekusi kedua secara independen
    re_success, re_msg = replay_and_verify_notebook(res.notebook_path, run_dir=run_dir)
    print(f"V6 Re-eksekusi Bersih: {re_success} - {re_msg}")
    assert re_success is True, f"Re-eksekusi notebook gagal: {re_msg}"
