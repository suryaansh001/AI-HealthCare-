"""Input validation and feature-family selection for ANCHOR model v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_MAP = {
    "yes": 1,
    "true": 1,
    "1": 1,
    "no": 0,
    "false": 0,
    "0": 0,
}
META_COLUMNS = {"label_num", "subid", "dttm_label", "lapse", "y", "hours_since_start"}


def encode_binary_target(values: pd.Series) -> pd.Series:
    """Map known binary labels and reject unknown or missing labels."""
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.isna() | ~normalized.isin(LABEL_MAP)
    if invalid.any():
        examples = normalized[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"Unknown or missing lapse labels: {examples}")
    return normalized.map(LABEL_MAP).astype("int8")


def load_feature_matrix(
    csv_path: str | Path,
    schema_path: str | Path,
    stride: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a feature matrix with strict identity, timestamp, and target checks."""
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    compact_features = schema.get("features")
    if not isinstance(compact_features, list) or not compact_features:
        raise ValueError("Feature schema must contain a non-empty 'features' list")

    frame = pd.read_csv(csv_path, low_memory=False)
    required = {"subid", "dttm_label", "lapse", *compact_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input matrix is missing required columns: {missing}")
    if frame["subid"].isna().any():
        raise ValueError("Participant identifiers must not be missing")

    frame["subid"] = frame["subid"].astype("string")
    frame["dttm_label"] = pd.to_datetime(frame["dttm_label"], utc=True, errors="coerce")
    if frame["dttm_label"].isna().any():
        raise ValueError("Every dttm_label must parse as a timestamp")
    if frame.duplicated(["subid", "dttm_label"]).any():
        raise ValueError("Duplicate participant/timestamp rows are not allowed")

    frame["y"] = encode_binary_target(frame["lapse"])
    frame = frame.sort_values(["subid", "dttm_label"], kind="stable").reset_index(drop=True)
    if stride > 1:
        keep = frame.groupby("subid", sort=False).cumcount().mod(stride).eq(0)
        frame = frame.loc[keep].reset_index(drop=True)
    return frame, schema


def build_feature_sets(
    columns: list[str],
    compact_features: list[str],
) -> dict[str, list[str]]:
    """Define consistent model-comparison groups from available columns."""
    predictors = [column for column in columns if column not in META_COLUMNS]
    historical = [
        column
        for column in predictors
        if column.startswith("p") and column.endswith(".count.lapse")
    ]
    ema = [column for column in predictors if column.startswith("ema_")]
    demographics = [column for column in predictors if column.startswith("demo_")]
    if not historical or not ema or not demographics:
        raise ValueError("Expected historical, EMA, and demographic feature families")

    sets = {
        "compact_22": list(compact_features),
        "historical_lapse_only": historical,
        "ema_only": ema,
        "demographic_only": demographics,
        "historical_plus_ema": historical + ema,
        "full_available": predictors,
    }
    for name, features in sets.items():
        if not features or len(features) != len(set(features)):
            raise ValueError(f"Feature set {name!r} is empty or contains duplicates")
        unavailable = sorted(set(features).difference(columns))
        if unavailable:
            raise ValueError(f"Feature set {name!r} references missing columns: {unavailable}")
    return sets


def feature_type_map(frame: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Return the input type contract used by training and inference."""
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot determine types for missing features: {missing}")
    result: dict[str, str] = {}
    for feature in features:
        dtype = frame[feature].dtype
        result[feature] = (
            "categorical"
            if (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            )
            else "numeric"
        )
    return result
