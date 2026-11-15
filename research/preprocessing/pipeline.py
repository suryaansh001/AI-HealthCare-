"""
research/preprocessing/pipeline.py
Leakage-safe preprocessing and feature family builders for Anchor ML.
Implements fold-safe transformations, categorical encoding, and feature selection.
"""
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

META_COLS = {"label_num", "subid", "dttm_label", "lapse", "y", "hours_since_start"}

PAT = re.compile(
    r"^(?:ema_(?P<item>\d+)\.)?p(?P<period>\d+)\.l(?P<lag>\d+)\."
    r"(?P<kind>[rd])(?P<stat>ratecount|recent|min|max|median)"
    r"(?:_response|\.count\.(?P<target>lapse|ema))$"
)


def get_feature_family(col: str) -> str:
    """Classify column name into documented feature family."""
    if col in META_COLS:
        return "meta"
    if col.startswith("demo_") or col == "label_day":
        return "static"
    m = PAT.match(col)
    if not m:
        return "unparsed"
    d = m.groupdict()
    kind = "change" if d["kind"] == "d" else "raw"
    if d["stat"] == "ratecount":
        return "past_use" if d["target"] == "lapse" else "compliance"
    if d["stat"] == "recent":
        return f"current_{kind}"
    return f"aggregate_{kind}"


def build_feature_sets(columns: List[str]) -> Dict[str, List[str]]:
    """Build named feature subsets from raw columns."""
    families = {c: get_feature_family(c) for c in columns}
    
    static_cols = [c for c, f in families.items() if f == "static"]
    past_use_cols = [c for c, f in families.items() if f == "past_use"]
    current_raw_cols = [c for c, f in families.items() if f == "current_raw"]
    current_change_cols = [c for c, f in families.items() if f == "current_change"]
    aggregate_raw_cols = [c for c, f in families.items() if f == "aggregate_raw"]
    aggregate_change_cols = [c for c, f in families.items() if f == "aggregate_change"]
    compliance_cols = [c for c, f in families.items() if f == "compliance"]

    base_c = static_cols + past_use_cols
    
    # Compact mobile checkin set (Items 2, 6, 7, 10 + past use + static)
    compact_ema = [c for c in current_raw_cols if any(k in c for k in ["ema_2", "ema_6", "ema_7", "ema_10"])]
    compact_set = static_cols + past_use_cols + compact_ema

    sets = {
        "A_static": static_cols,
        "B_past_use": past_use_cols,
        "C_static_past_use": base_c,
        "D_C_current_raw": base_c + current_raw_cols,
        "E_C_raw_state": base_c + current_raw_cols + aggregate_raw_cols,
        "F_C_change_only": base_c + current_change_cols + aggregate_change_cols,
        "G_full_features": base_c + current_raw_cols + current_change_cols + aggregate_raw_cols + aggregate_change_cols,
        "H_all_compliance": base_c + compliance_cols + current_raw_cols + current_change_cols + aggregate_raw_cols + aggregate_change_cols,
        "K_compact_mobile": compact_set
    }
    return sets


def make_fold_safe_pipeline(
    feature_columns: List[str],
    df_sample: pd.DataFrame,
    model_type: str = "hgb",
    estimator: Optional[BaseEstimator] = None
) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with fold-safe preprocessing.
    Categorical columns are one-hot encoded without leakage.
    Numeric columns are imputed and scaled if using linear/distance models.
    """
    cat_cols = [c for c in feature_columns if c in df_sample.columns and (
        df_sample[c].dtype == object or str(df_sample[c].dtype).startswith("str")
    )]
    num_cols = [c for c in feature_columns if c not in cat_cols and c in df_sample.columns]

    transformers = []
    if cat_cols:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    if model_type == "logit":
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_transformer, num_cols))
    else:
        # HistGradientBoosting handles NaNs natively
        num_transformer = "passthrough"
        transformers.append(("num", num_transformer, num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline_steps = [("preprocessor", preprocessor)]
    if estimator is not None:
        pipeline_steps.append(("model", estimator))

    return Pipeline(pipeline_steps)
