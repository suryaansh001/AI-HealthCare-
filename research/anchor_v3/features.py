"""Leakage-conscious, past-sample-only participant feature transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EMA_FEATURES = [
    "ema_2.p0.l24.rrecent_response",
    "ema_6.p0.l24.rrecent_response",
    "ema_7.p0.l24.rrecent_response",
    "ema_10.p0.l24.rrecent_response",
]
RECENT_LAPSE_COUNT = "p24.l24.rratecount.count.lapse"
LONG_LAPSE_COUNT = "p168.l24.rratecount.count.lapse"


def add_personal_history_features(
    frame: pd.DataFrame,
    window_samples: int = 24,
    minimum_history: int = 6,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create lagged per-person EMA baselines and nominal short/long count contrasts.

    The rolling transforms use at most the preceding `window_samples` rows of that
    participant and exclude the current row. A row is not a fixed duration; timestamps
    and the source feature timing remain subject to the study's unresolved contract.
    """
    if window_samples < 2 or minimum_history < 2 or minimum_history > window_samples:
        raise ValueError("Require 2 <= minimum_history <= window_samples")
    required = {"subid", "dttm_label", *EMA_FEATURES, RECENT_LAPSE_COUNT, LONG_LAPSE_COUNT}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot build personalization features; missing: {missing}")
    if frame["subid"].isna().any() or frame["dttm_label"].isna().any():
        raise ValueError("Participant ID and prediction timestamp must be present")
    ordered = frame.sort_values(["subid", "dttm_label"], kind="stable")
    if not ordered.index.equals(frame.index):
        raise ValueError("Input must already be sorted by participant and timestamp")
    if frame.duplicated(["subid", "dttm_label"]).any():
        raise ValueError("Duplicate participant/timestamp rows are not allowed")

    result = frame.copy()
    result["v3_prior_sample_count"] = result.groupby("subid", sort=False).cumcount()
    personal_ema_features: list[str] = []
    missingness_features: list[str] = []
    for column in EMA_FEATURES:
        values = pd.to_numeric(result[column], errors="coerce")
        previous_mean = result.assign(_value=values).groupby("subid", sort=False)["_value"].transform(
            lambda series: series.shift(1).rolling(
                window=window_samples,
                min_periods=minimum_history,
            ).mean()
        )
        previous_std = result.assign(_value=values).groupby("subid", sort=False)["_value"].transform(
            lambda series: series.shift(1).rolling(
                window=window_samples,
                min_periods=minimum_history,
            ).std()
        )
        previous_count = result.assign(_value=values).groupby("subid", sort=False)["_value"].transform(
            lambda series: series.shift(1).rolling(
                window=window_samples,
                min_periods=1,
            ).count()
        )
        previous_missing_rate = result.assign(_missing=values.isna().astype(float)).groupby(
            "subid", sort=False
        )["_missing"].transform(
            lambda series: series.shift(1).rolling(
                window=window_samples,
                min_periods=minimum_history,
            ).mean()
        )

        short_name = column.split(".")[0]
        mean_name = f"v3_{short_name}_prior{window_samples}_mean"
        deviation_name = f"v3_{short_name}_deviation_from_prior{window_samples}"
        zscore_name = f"v3_{short_name}_standardized_deviation"
        count_name = f"v3_{short_name}_prior{window_samples}_observed_count"
        missing_name = f"v3_{short_name}_prior{window_samples}_missing_rate"
        current_missing_name = f"v3_{short_name}_current_missing"

        result[mean_name] = previous_mean
        result[deviation_name] = values - previous_mean
        valid_std = previous_std.where(previous_std > 1e-8)
        result[zscore_name] = ((values - previous_mean) / valid_std).clip(-10, 10)
        result[count_name] = previous_count.astype(float)
        result[missing_name] = previous_missing_rate
        result[current_missing_name] = values.isna().astype("int8")
        personal_ema_features.extend(
            [mean_name, deviation_name, zscore_name, count_name]
        )
        missingness_features.extend([missing_name, current_missing_name])

    recent_count = pd.to_numeric(result[RECENT_LAPSE_COUNT], errors="coerce")
    long_count = pd.to_numeric(result[LONG_LAPSE_COUNT], errors="coerce")
    recent_rate = recent_count / 24.0
    long_rate = long_count / 168.0
    result["v3_nominal_lapse_rate_24_minus_168"] = recent_rate - long_rate
    result["v3_nominal_lapse_rate_ratio_24_to_168"] = (
        (recent_rate + 1 / 24) / (long_rate + 1 / 168)
    ).replace([np.inf, -np.inf], np.nan)

    history_rate_features = [
        "v3_nominal_lapse_rate_24_minus_168",
        "v3_nominal_lapse_rate_ratio_24_to_168",
    ]
    personal_ema_features.extend(missingness_features)
    personal_all_features = personal_ema_features + history_rate_features
    families: dict[str, list[str]] = {
        "personal_ema_baseline": personal_ema_features,
        "history_rate_contrast": history_rate_features,
        "personalized_hybrid": personal_all_features,
    }
    return result, families


def assert_future_independent(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    feature_columns: list[str],
    cut_position: int,
) -> None:
    """Raise if mutating later feature rows changed any earlier derived values."""
    if len(original) != len(transformed):
        raise ValueError("Original and transformed frames must have equal row counts")
    if not 0 <= cut_position <= len(original):
        raise ValueError("cut_position is outside the frame")
    changed = original.copy()
    if cut_position < len(changed):
        changed.loc[changed.index[cut_position:], feature_columns] = 999999
    altered, _ = add_personal_history_features(changed)
    derived = [
        column
        for column in transformed.columns
        if column.startswith("v3_")
    ]
    before = transformed.iloc[:cut_position][derived].reset_index(drop=True)
    after = altered.iloc[:cut_position][derived].reset_index(drop=True)
    if not before.equals(after):
        raise AssertionError("Future feature values changed a prior row's engineered features")
