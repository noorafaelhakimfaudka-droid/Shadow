# shadow_core/modeling.py
"""
S6 — Eksperimen Terkontrol (PRD 6.6 & Aturan [K]).
Mengevaluasi baseline dan kandidat model pada lipatan identik (Aturan AC-03).
Mencatat setiap percobaan ke ledger (Aturan R5).
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, TimeSeriesSplit

from shadow_core.ledger import Ledger
from shadow_core.pipeline import build_preprocessing_pipeline, create_model_pipeline, hash_pipeline_config, PipelineConfig


@dataclass
class ModelCandidate:
    name: str
    estimator: Any
    requires_scaling: bool = False
    complexity: int = 1


def generate_folds(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    task_type: str,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Menghasilkan partisi lipatan CV dan hash definisi lipatan."""
    y_arr = np.array(y)
    n_samples = len(X)

    if time_col is not None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
    elif task_type in ("binary_classification", "multiclass_classification"):
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        splits = list(rskf.split(X, y_arr))
    else:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        splits = list(rkf.split(X, y_arr))

    # Serialisasi indeks lipatan untuk hash integritas
    fold_repr = []
    for train_idx, val_idx in splits:
        fold_repr.append({
            "train": train_idx.tolist()[:10] + [len(train_idx)],
            "val": val_idx.tolist()[:10] + [len(val_idx)],
        })

    fold_hash = hashlib.sha256(json.dumps(fold_repr).encode("utf-8")).hexdigest()
    return splits, fold_hash


def evaluate_candidate_on_folds(
    candidate: ModelCandidate,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    primary_metric: str,
    task_type: str,
    pipe_config: Optional[PipelineConfig] = None,
) -> Tuple[List[float], Dict[str, Any], float, int, str]:
    """Mengevaluasi satu model kandidat pada lipatan identik dengan fit-in-fold."""
    y_arr = np.array(y)
    scores = []
    start_time = time.time()
    ram_start = psutil.Process().memory_info().rss // (1024 * 1024)

    # Buat preprocessing pipeline
    if pipe_config is None:
        pipe_config = PipelineConfig(numeric_scaler="standard" if candidate.requires_scaling else None)
    else:
        if candidate.requires_scaling and pipe_config.numeric_scaler is None:
            pipe_config.numeric_scaler = "standard"

    preprocessor, _, _ = build_preprocessing_pipeline(X, pipe_config)
    full_pipe = create_model_pipeline(preprocessor, candidate.estimator)
    pipe_hash = hash_pipeline_config(full_pipe)

    for train_idx, val_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = clone(full_pipe)
        model.fit(X_train, y_train)

        # Evaluasi
        if primary_metric == "pr_auc" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs
            score = float(average_precision_score(y_val, p1))
        elif primary_metric == "balanced_accuracy":
            preds = model.predict(X_val)
            score = float(balanced_accuracy_score(y_val, preds))
        elif primary_metric in ("macro_f1", "f1"):
            preds = model.predict(X_val)
            avg = "macro" if task_type == "multiclass_classification" else "binary"
            score = float(f1_score(y_val, preds, average=avg))
        elif primary_metric == "mae":
            preds = model.predict(X_val)
            score = float(mean_absolute_error(y_val, preds))
        elif primary_metric == "rmse":
            preds = model.predict(X_val)
            score = float(root_mean_squared_error(y_val, preds))
        elif primary_metric == "r2":
            preds = model.predict(X_val)
            score = float(r2_score(y_val, preds))
        else:
            preds = model.predict(X_val)
            score = float(balanced_accuracy_score(y_val, preds)) if task_type != "regression" else float(mean_absolute_error(y_val, preds))

        scores.append(round(score, 5))

    elapsed = round(time.time() - start_time, 2)
    ram_peak = max(ram_start, psutil.Process().memory_info().rss // (1024 * 1024))

    mean_score = float(np.mean(scores))
    sd_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    se_score = sd_score / np.sqrt(len(scores))
    ci95 = [round(mean_score - 1.96 * se_score, 4), round(mean_score + 1.96 * se_score, 4)]

    summary = {
        "mean": round(mean_score, 4),
        "sd": round(sd_score, 4),
        "ci95": ci95,
        "n_folds": len(scores),
    }

    return scores, summary, elapsed, ram_peak, pipe_hash


def get_default_candidates(task_type: str, seed: int = 42) -> List[ModelCandidate]:
    """Menghasilkan kandidat model default sesuai PRD 6.6."""
    if task_type == "regression":
        return [
            ModelCandidate("Dummy", DummyRegressor(strategy="median"), complexity=0),
            ModelCandidate("Linear", Ridge(alpha=1.0), requires_scaling=True, complexity=1),
            ModelCandidate("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1), complexity=2),
            ModelCandidate("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=100, random_state=seed), complexity=3),
        ]
    else:
        return [
            ModelCandidate("Dummy", DummyClassifier(strategy="prior"), complexity=0),
            ModelCandidate("Linear", LogisticRegression(max_iter=1000, random_state=seed), requires_scaling=True, complexity=1),
            ModelCandidate("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1), complexity=2),
            ModelCandidate("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100, random_state=seed), complexity=3),
        ]


def run_controlled_experiments(
    X_dev: pd.DataFrame,
    y_dev: pd.Series | np.ndarray,
    task_type: str,
    primary_metric: str,
    run_id: str,
    case_id: str,
    ledger: Ledger,
    seed: int = 42,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> Tuple[Dict[str, List[float]], str, Dict[str, Any]]:
    """Menjalankan seluruh baseline dan kandidat pada lipatan identik dan mencatat ke ledger."""
    splits, fold_hash = generate_folds(
        X=X_dev,
        y=y_dev,
        task_type=task_type,
        time_col=time_col,
        group_col=group_col,
        seed=seed,
    )

    candidates = get_default_candidates(task_type, seed=seed)
    candidate_scores: Dict[str, List[float]] = {}
    candidate_summaries: Dict[str, Any] = {}

    for cand in candidates:
        scores, summary, elapsed, ram_peak, pipe_hash = evaluate_candidate_on_folds(
            candidate=cand,
            X=X_dev,
            y=y_dev,
            splits=splits,
            primary_metric=primary_metric,
            task_type=task_type,
        )

        candidate_scores[cand.name] = scores
        candidate_summaries[cand.name] = summary

        # Catat ke ledger (R5)
        ledger_entry = {
            "trial_id": "",  # Di-generate otomatis oleh ledger
            "run_id": run_id,
            "case_id": case_id,
            "ts": "",
            "agent": "core_experimenter",
            "hipotesis": f"Evaluasi model kandidat {cand.name}",
            "model": cand.name,
            "hiperparameter": cand.estimator.get_params() if hasattr(cand.estimator, "get_params") else {},
            "pipeline_hash": pipe_hash,
            "fold_def_hash": fold_hash,
            "seed": seed,
            "skor_per_lipatan": scores,
            "ringkasan": summary,
            "waktu_detik": elapsed,
            "ram_puncak_mb": ram_peak,
            "status": "selesai",
            "alasan": None,
            "cell_notebook": "8.1",
            "akses_uji_tersegel": False,
        }
        ledger.append(ledger_entry)

    return candidate_scores, fold_hash, candidate_summaries
