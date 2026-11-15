"""Validation for raw-source structure and exact model-ready feature frames."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from research.anchor_features.schema import FeatureSchema


IDENTITY_COLUMNS = ("subid", "dttm_label")
RAW_EMA_REQUIRED = ("subid", "dttm_obs", "ema_2", "ema_6", "ema_7", "ema_10")


class UnresolvedFeatureSourceError(ValueError):
    """Raised when raw inputs cannot support a verified model feature transform."""


def normalize_identity(
    frame: pd.DataFrame,
    participant_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    required = {participant_column, timestamp_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing identity columns: {missing}")
    result = frame[[participant_column, timestamp_column]].copy()
    if result[participant_column].isna().any():
        raise ValueError(f"{participant_column} must not be missing")
    result[participant_column] = result[participant_column].astype("string")
    if result[participant_column].str.strip().eq("").any():
        raise ValueError(f"{participant_column} must not be empty")
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        utc=True,
        errors="coerce",
    )
    if result[timestamp_column].isna().any():
        raise ValueError(f"{timestamp_column} contains missing or invalid timestamps")
    if result.duplicated([participant_column, timestamp_column]).any():
        raise ValueError(
            f"Duplicate ({participant_column}, {timestamp_column}) observations found"
        )
    return result


def validate_raw_ema_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw EMA keys and target-schema item columns without deriving features."""
    missing = sorted(set(RAW_EMA_REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Raw EMA source is missing required columns: {missing}")
    identity = normalize_identity(frame, "subid", "dttm_obs")
    result = frame.copy()
    result["subid"] = identity["subid"]
    result["dttm_obs"] = identity["dttm_obs"]
    for column in ("ema_2", "ema_6", "ema_7", "ema_10"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & numeric.isna()
        if invalid.any():
            raise TypeError(f"Raw EMA response {column!r} contains non-numeric values")
        finite = numeric.dropna().map(lambda value: math.isfinite(float(value)))
        if not finite.all():
            raise ValueError(f"Raw EMA response {column!r} contains non-finite values")
        result[column] = numeric.astype("float64")
    return result.sort_values(["subid", "dttm_obs"], kind="stable").reset_index(
        drop=True
    )


def validate_feature_mapping(
    record: Mapping[str, Any],
    schema: FeatureSchema,
) -> dict[str, Any]:
    """Validate exact feature keys, types, finite values, and trained categories."""
    if not isinstance(record, Mapping):
        raise TypeError("Feature record must be a mapping")
    if any(not isinstance(name, str) for name in record):
        raise TypeError("Feature names must be strings")
    expected = set(schema.features)
    actual = set(record)
    missing = [name for name in schema.features if name not in actual]
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")

    validated: dict[str, Any] = {}
    for name in schema.features:
        value = record[name]
        if value is None or value is pd.NA or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value))
        ):
            validated[name] = None
            continue
        if schema.feature_types[name] == "numeric":
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Real
            ):
                raise TypeError(f"Feature {name!r} must be numeric or null")
            if not math.isfinite(float(value)):
                raise ValueError(f"Feature {name!r} must be finite or null")
            validated[name] = float(value)
        else:
            if not isinstance(value, str):
                raise TypeError(f"Feature {name!r} must be a string or null")
            if value not in schema.categorical_values[name]:
                raise ValueError(
                    f"Feature {name!r} has an unknown category: {value!r}"
                )
            validated[name] = value
    return validated


def validate_model_ready_frame(
    frame: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Require exactly the 22 model fields in trained order; preserve nulls."""
    missing = [name for name in schema.features if name not in frame.columns]
    extra = [name for name in frame.columns if name not in schema.features]
    if missing or extra:
        raise ValueError(f"Model-ready columns mismatch; missing={missing}, extra={extra}")

    result = frame.loc[:, list(schema.features)].copy()
    for name in schema.features:
        if schema.feature_types[name] == "numeric":
            original = result[name]
            boolean = original.map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if boolean.any():
                raise TypeError(f"Feature {name!r} contains boolean values")
            numeric = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                raise TypeError(f"Feature {name!r} contains non-numeric values")
            finite = numeric.dropna().map(lambda value: math.isfinite(float(value)))
            if not finite.all():
                raise ValueError(f"Feature {name!r} contains non-finite values")
            result[name] = numeric.astype("float64")
        else:
            values = result[name]
            invalid_type = values.notna() & ~values.map(lambda value: isinstance(value, str))
            if invalid_type.any():
                raise TypeError(f"Feature {name!r} contains non-string categories")
            invalid_category = values.notna() & ~values.isin(
                schema.categorical_values[name]
            )
            if invalid_category.any():
                unknown = sorted(values[invalid_category].astype(str).unique().tolist())
                raise ValueError(f"Feature {name!r} contains unknown categories: {unknown}")
            result[name] = values.astype("string")
    return result


def require_verified_raw_feature_generation(
    requested_features: tuple[str, ...],
) -> NoReturn:
    """Fail closed: authoritative raw-to-schema transforms are not in this checkout."""
    raise UnresolvedFeatureSourceError(
        "Raw-to-model feature generation is not verified. The exact historical "
        "event source/windows, demographic mapping, EMA item-response encoding, "
        "and l24 timestamp alignment are unavailable. Use the existing "
        "engineered 24-lag matrix as a pass-through source; unresolved schema "
        f"features: {list(requested_features)}"
    )
