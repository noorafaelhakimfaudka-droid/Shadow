# shadow_core/vault_process.py
"""
S3b — Vault Process: Proses Brankas Terisolasi (PRD 5.3 & 6.3 S3a.3).
Satu-satunya komponen yang memegang kunci untuk membuka sealed.bin.
Menerapkan batasan akses tepat 1x melalui marker disk vault.spent.
Panggilan kedua -> SealedAccessError.
"""

from __future__ import annotations
import io
import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


class SealedAccessError(Exception):
    """Pelanggaran Keamanan Brankas: Uji tersegel sudah pernah diakses atau akses tidak sah."""


def compute_metrics_and_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metric_names: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Menghitung metrik evaluasi dan rentang bootstrap 95% pada baris uji independen."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    def calc_metric(name: str, yt: np.ndarray, yp: np.ndarray, yprob: Optional[np.ndarray]) -> float:
        if name == "balanced_accuracy":
            return float(balanced_accuracy_score(yt, yp))
        elif name == "macro_f1" or name == "f1":
            return float(f1_score(yt, yp, average="macro" if len(np.unique(yt)) > 2 else "binary"))
        elif name == "pr_auc":
            if yprob is not None:
                # Binary: proba class 1
                prob = yprob[:, 1] if yprob.ndim == 2 and yprob.shape[1] >= 2 else yprob
                return float(average_precision_score(yt, prob))
            return 0.0
        elif name == "log_loss":
            if yprob is not None:
                return float(log_loss(yt, yprob))
            return 0.0
        elif name == "brier_score":
            if yprob is not None:
                prob = yprob[:, 1] if yprob.ndim == 2 and yprob.shape[1] >= 2 else yprob
                return float(brier_score_loss(yt, prob))
            return 0.0
        elif name == "mae":
            return float(mean_absolute_error(yt, yp))
        elif name == "rmse":
            return float(root_mean_squared_error(yt, yp))
        elif name == "r2":
            return float(r2_score(yt, yp))
        else:
            return float(mean_absolute_error(yt, yp))

    # 1. Point metrics
    point_metrics = {}
    for m in metric_names:
        try:
            point_metrics[m] = round(calc_metric(m, y_true, y_pred, y_proba), 5)
        except Exception:
            point_metrics[m] = 0.0

    # 2. Bootstrap 95% CI over rows
    bootstrap_ci = {}
    indices = np.arange(n_samples)
    for m in metric_names:
        boot_vals = []
        for _ in range(n_bootstrap):
            b_idx = rng.choice(indices, size=n_samples, replace=True)
            b_yt = y_true[b_idx]
            b_yp = y_pred[b_idx]
            b_prob = y_proba[b_idx] if y_proba is not None else None
            try:
                boot_vals.append(calc_metric(m, b_yt, b_yp, b_prob))
            except Exception:
                continue
        if len(boot_vals) >= 10:
            low, high = np.percentile(boot_vals, [2.5, 97.5])
            bootstrap_ci[m] = [round(float(low), 5), round(float(high), 5)]
        else:
            bootstrap_ci[m] = [point_metrics.get(m, 0.0), point_metrics.get(m, 0.0)]

    # 3. Calibration curve (jika klasifikasi probabilitas tersedia)
    calib_data = {}
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            prob = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] >= 2 else y_proba
            prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=5)
            calib_data = {
                "prob_true": [round(float(x), 4) for x in prob_true],
                "prob_pred": [round(float(x), 4) for x in prob_pred],
            }
        except Exception:
            pass

    return {
        "metrics": point_metrics,
        "ci95": bootstrap_ci,
        "calibration": calib_data,
    }


def execute_vault_evaluation(
    run_dir: Path | str,
    model_path: Path | str,
    target_col: str,
    metric_names: List[str],
) -> Dict[str, Any]:
    """Eksekusi internal dalam proses brankas terpisah."""
    run_d = Path(run_dir)
    spent_file = run_d / "vault.spent"

    if spent_file.exists():
        raise SealedAccessError("Pelanggaran Keamanan Brankas: Uji tersegel sudah pernah diakses!")

    key_file = run_d / "vault.key"
    sealed_file = run_d / "sealed.bin"

    if not sealed_file.exists():
        raise FileNotFoundError(f"File sealed.bin tidak ditemukan di {run_d}")

    # Cek jika data kecil (sealed kosong)
    if sealed_file.stat().st_size == 0:
        return {
            "metrics": {},
            "ci95": {},
            "calibration": {},
            "is_small_data": True,
            "note": "Uji tersegel tidak dibuat karena data kecil (menggunakan Nested CV).",
        }

    if not key_file.exists():
        raise FileNotFoundError(f"Kunci vault.key tidak ditemukan di {run_d}")

    # Buka brankas
    key = key_file.read_bytes()
    fernet = Fernet(key)
    encrypted_bytes = sealed_file.read_bytes()
    parquet_bytes = fernet.decrypt(encrypted_bytes)

    df_sealed = pd.read_parquet(io.BytesIO(parquet_bytes))
    X_sealed = df_sealed.drop(columns=[target_col])
    y_sealed = df_sealed[target_col].to_numpy()

    # Muat model
    model = joblib.load(model_path)
    if hasattr(model, "feature_names_in_"):
        cols_to_use = [c for c in model.feature_names_in_ if c in X_sealed.columns]
        X_sealed = X_sealed[cols_to_use]

    y_pred = model.predict(X_sealed)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_sealed)
        except Exception:
            y_proba = None

    result = compute_metrics_and_bootstrap(
        y_true=y_sealed,
        y_pred=y_pred,
        y_proba=y_proba,
        metric_names=metric_names,
    )

    # Tulis tanda disk vault.spent
    spent_payload = {
        "spent_at": datetime.now(timezone.utc).isoformat(),
        "model_file": str(Path(model_path).name),
        "metrics_summary": result["metrics"],
    }
    spent_file.write_text(json.dumps(spent_payload, indent=2), encoding="utf-8")

    return result


class VaultClient:
    """Klien brankas yang dipanggil oleh pipeline utama."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)

    @property
    def access_count(self) -> int:
        """Menghitung berapa kali brankas diakses (dibaca langsung dari disk)."""
        spent_file = self.run_dir / "vault.spent"
        return 1 if spent_file.exists() else 0

    def evaluate_once(
        self,
        model_path: Path | str,
        target_col: str,
        metric_names: List[str],
    ) -> Dict[str, Any]:
        """Mengevaluasi model pada uji tersegel di subproses terpisah."""
        spent_file = self.run_dir / "vault.spent"
        if spent_file.exists():
            raise SealedAccessError("Pelanggaran Keamanan Brankas: Data uji tersegel sudah pernah diakses!")

        # Simpan request evaluasi
        req_file = self.run_dir / "vault_req.json"
        out_file = self.run_dir / "vault_out.json"

        req_data = {
            "run_dir": str(self.run_dir.resolve()),
            "model_path": str(Path(model_path).resolve()),
            "target_col": target_col,
            "metric_names": metric_names,
            "out_file": str(out_file.resolve()),
        }
        req_file.write_text(json.dumps(req_data), encoding="utf-8")

        # Jalankan di subprocess terpisah
        cmd = [sys.executable, "-m", "shadow_core.vault_process", str(req_file.resolve())]
        log_file = self.run_dir / "vault_proc.log"
        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                res = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)

            if res.returncode != 0:
                err_msg = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
                if "SealedAccessError" in err_msg or res.returncode == 42:
                    raise SealedAccessError(f"Proses brankas menolak: {err_msg}")
                raise RuntimeError(f"Gagal mengeksekusi evaluasi brankas (kode {res.returncode}): {err_msg}")

            if not out_file.exists():
                raise RuntimeError("Hasil evaluasi brankas tidak ditemukan!")

            result = json.loads(out_file.read_text(encoding="utf-8"))
            req_file.unlink(missing_ok=True)
            out_file.unlink(missing_ok=True)
            log_file.unlink(missing_ok=True)
            return result

        except OSError as os_err:
            # Fallback aman jika OS memblokir DuplicateHandle pipa di thread/lingkungan Windows
            return execute_vault_evaluation(
                run_dir=self.run_dir,
                model_path=model_path,
                target_col=target_col,
                metric_names=metric_names,
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    req_path = Path(sys.argv[1])
    data = json.loads(req_path.read_text(encoding="utf-8"))

    try:
        eval_result = execute_vault_evaluation(
            run_dir=data["run_dir"],
            model_path=data["model_path"],
            target_col=data["target_col"],
            metric_names=data["metric_names"],
        )
        Path(data["out_file"]).write_text(json.dumps(eval_result), encoding="utf-8")
        sys.exit(0)
    except SealedAccessError as sae:
        sys.stderr.write(f"SealedAccessError: {sae}\n")
        sys.exit(42)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
