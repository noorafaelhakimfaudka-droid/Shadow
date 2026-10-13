# tests/test_v8_source_integrity.py
"""
Verifikasi V8: Integritas Berkas Sumber (PRD 10.1 & Aturan R9).
Target [K]: 0 pelanggaran integritas sumber (hash sebelum == sesudah) pada 100% run.
"""

import pytest
from pathlib import Path
import pandas as pd
from shadow_core.runner import run_case
from shadow_core.manifest import hash_file
from shadow_eval.seed_defects import generate_clean_dataset


def test_v8_source_file_never_modified(tmp_path: Path):
    csv_file = tmp_path / "source_data.csv"
    df = generate_clean_dataset(n_samples=200, task="binary", seed=42)
    df.to_csv(csv_file, index=False)

    hash_before = hash_file(csv_file)

    run_dir = tmp_path / "run_v8"
    res = run_case(
        csv_path=csv_file,
        target_col="target",
        goal="Uji integritas berkas sumber",
        run_dir=run_dir,
    )

    hash_after = hash_file(csv_file)
    print(f"\nV8 Hash Sebelum: {hash_before}")
    print(f"V8 Hash Sesudah: {hash_after}")

    assert hash_before == hash_after, "File CSV sumber telah dimutasi selama eksekusi!"
    assert res.manifest["input"]["sha256"] == hash_after
    assert res.manifest["input"]["sha256_sesudah"] == hash_after
