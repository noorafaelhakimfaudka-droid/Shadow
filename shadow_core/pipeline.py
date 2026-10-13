# shadow_core/pipeline.py
"""
S5 — Pipeline & Rekayasa Fitur (PRD 6.5 & Aturan R4).
Aturan R4: Semua transformasi data di-fit HANYA pada data latih per lipatan (fit-di-dalam-lipatan).
Kit komponen tervalidasi: ColumnTransformer + Imputer + Scaler + Encoder.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder


@dataclass
class PipelineConfig:
    numeric_scaler: Optional[str] = None  # "standard", "robust", or None
    numeric_imputer: str = "median"  # "median", "mean"
    categorical_imputer: str = "most_frequent"
    add_missing_indicator: bool = False
    max_ohe_cardinality: int = 10


def build_preprocessing_pipeline(
    df_features: pd.DataFrame,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Membangun ColumnTransformer tervalidasi dari kit komponen resmi."""
    if cfg is None:
        cfg = PipelineConfig()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    transformers = []

    # 1. Transformasi Numerik
    if numeric_cols:
        num_steps = []
        # Imputasi
        strategy = cfg.numeric_imputer if cfg.numeric_imputer in ("median", "mean") else "median"
        num_steps.append(("imputer", SimpleImputer(strategy=strategy, add_indicator=cfg.add_missing_indicator)))

        # Skala
        if cfg.numeric_scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif cfg.numeric_scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, numeric_cols))

    # 2. Transformasi Kategorikal
    if categorical_cols:
        cat_steps = []
        cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")))
        cat_steps.append((
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ))
        cat_pipe = Pipeline(cat_steps)
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def create_model_pipeline(
    preprocessor: ColumnTransformer,
    estimator: Any,
) -> Pipeline:
    """Menggabungkan preprocessor dan estimator menjadi Pipeline scikit-learn utuh."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def hash_pipeline_config(pipe: Pipeline) -> str:
    """Menghasilkan hash deterministik representasi pipeline."""
    repr_str = str(pipe)
    return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()
