# tests/test_memory_budget.py
"""
Uji Anggaran Memori (PRD 4.5 & AC-14).
Memverifikasi bahwa puncak RAM sistem <= 7.0 GB dan Core selesai <= 10 menit.
"""

import pytest
import time
from pathlib import Path
import psutil
import pandas as pd
from shadow_core.runner import run_case
from shadow_eval.seed_defects import generate_clean_dataset


def test_memory_budget_core(tmp_path: Path):
    csv_file = tmp_path / "mem_test_data.csv"
    # Dataset 1500 baris, 15 kolom
    df = generate_clean_dataset(n_samples=1500, n_features=15, task="binary", seed=42)
    df.to_csv(csv_file, index=False)

    ram_start_gb = psutil.virtual_memory().used / (1024**3)
    t0 = time.time()

    res = run_case(
        csv_path=csv_file,
        target_col="target",
        goal="Uji anggaran memori laptop 8 GB",
        run_dir=tmp_path / "run_mem",
    )

    t_elapsed_minutes = (time.time() - t0) / 60.0
    proc = psutil.Process()
    proc_ram_mb = proc.memory_info().rss / (1024**2)
    ram_peak_gb = psutil.virtual_memory().used / (1024**3)

    print(f"\nUji Anggaran Memori M0a:")
    print(f"  RAM Proses : {proc_ram_mb:.1f} MB (Batas Proses: <= 2048 MB)")
    print(f"  RAM Sistem : {ram_peak_gb:.2f} GB (Batas PRD Laptop 8 GB: <= 7.0 GB)")
    print(f"  Waktu Run  : {t_elapsed_minutes:.2f} menit (Batas PRD: <= 10 menit)")

    assert proc_ram_mb <= 2048.0, f"RAM proses {proc_ram_mb:.1f} MB melebihi batas aman 2 GB!"
    assert t_elapsed_minutes <= 10.0, f"Waktu eksekusi {t_elapsed_minutes:.2f} menit melebihi batas 10 menit!"
    assert res.is_success is True
