# tests/test_vault.py
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from shadow_core.splitting import (
    pisah_brankas,
    detect_structure_without_target,
    DevData,
)
from shadow_core.vault_process import VaultClient, SealedAccessError


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "dataset.csv"
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n),
        "feature2": np.random.uniform(10, 50, n),
        "user_id": np.random.choice([f"user_{i}" for i in range(100)], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_vault_split_and_single_access(tmp_path: Path, sample_csv: Path):
    run_dir = tmp_path / "run_01"

    # Pisahkan brankas
    dev_data = pisah_brankas(
        csv_path=sample_csv,
        target_col="target",
        run_dir=run_dir,
        seed=42,
        sealed_fraction=0.20,
    )

    assert isinstance(dev_data, DevData)
    assert dev_data.dev_df is not None
    assert dev_data.n_dev + dev_data.n_sealed == 1000
    assert 150 <= dev_data.n_sealed <= 250
    assert (run_dir / "dev.parquet").exists()
    assert (run_dir / "sealed.bin").exists()
    assert (run_dir / "vault.key").exists()

    # Latih model sederhana pada dev data
    X_dev = dev_data.dev_df[["feature1", "feature2"]]
    y_dev = dev_data.dev_df["target"]
    model = LogisticRegression().fit(X_dev, y_dev)

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)

    vault_client = VaultClient(run_dir)
    assert vault_client.access_count == 0

    # Akses 1: harus berhasil
    result = vault_client.evaluate_once(
        model_path=model_path,
        target_col="target",
        metric_names=["balanced_accuracy", "pr_auc"],
    )

    assert "metrics" in result
    assert "balanced_accuracy" in result["metrics"]
    assert "ci95" in result
    assert vault_client.access_count == 1
    assert (run_dir / "vault.spent").exists()

    # V9a: Akses 2: harus gagal dengan SealedAccessError!
    with pytest.raises(SealedAccessError):
        vault_client.evaluate_once(
            model_path=model_path,
            target_col="target",
            metric_names=["balanced_accuracy"],
        )

    # Re-instantiate VaultClient (mensimulasikan restart proses)
    new_client = VaultClient(run_dir)
    assert new_client.access_count == 1
    with pytest.raises(SealedAccessError):
        new_client.evaluate_once(
            model_path=model_path,
            target_col="target",
            metric_names=["balanced_accuracy"],
        )


def test_v9c_target_blind_splitting(tmp_path: Path):
    """V9c: Target-blind splitting: Mengacak target tidak mengubah deteksi struktur."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "signup_date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "amount": np.random.uniform(10, 100, n),
        "target": np.random.choice([0, 1], n),
    })

    t_col1, g_col1, strat1 = detect_structure_without_target(df, "target")

    # Acak target sepenuhnya
    df["target"] = np.random.permutation(df["target"].values)
    t_col2, g_col2, strat2 = detect_structure_without_target(df, "target")

    assert t_col1 == t_col2 == "signup_date"
    assert g_col1 == g_col2 is None
    assert strat1 == strat2
