"""Strict, research-only inference for a versioned ANCHOR v2 artifact."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a trusted joblib artifact; joblib files must not come from untrusted sources."""
    artifact = joblib.load(path)
    required = {"model_version", "model", "feature_schema", "target_status"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Artifact is missing required metadata: {sorted(missing)}")
    if artifact["target_status"] != "NOT_VERIFIED":
        raise ValueError("Unexpected target status; inspect the artifact contract")
    schema = artifact["feature_schema"]
    features = schema.get("features") if isinstance(schema, dict) else None
    feature_types = schema.get("feature_types") if isinstance(schema, dict) else None
    if (
        not isinstance(features, list)
        or not features
        or len(features) != len(set(features))
        or not isinstance(feature_types, dict)
        or set(feature_types) != set(features)
        or any(kind not in {"numeric", "categorical"} for kind in feature_types.values())
    ):
        raise ValueError("Artifact feature schema or type declarations are invalid")
    if not callable(getattr(artifact["model"], "predict_proba", None)):
        raise ValueError("Artifact model does not provide predict_proba")
    return artifact


def validate_record(
    record: Mapping[str, Any],
    features: list[str],
    feature_types: dict[str, str],
) -> pd.DataFrame:
    """Require the exact feature keys and compatible values; null values are imputed."""
    if not isinstance(record, Mapping):
        raise TypeError("Inference input must be a mapping of feature names to values")
    expected = set(features)
    actual = set(record)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Input schema mismatch; missing={missing}, extra={extra}")

    validated: dict[str, Any] = {}
    for feature in features:
        value = record[feature]
        if value is None or bool(pd.isna(value)):
            validated[feature] = value
            continue
        kind = feature_types.get(feature)
        if kind == "numeric":
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"Feature {feature!r} must be numeric or null")
            if not math.isfinite(float(value)):
                raise ValueError(f"Feature {feature!r} must be finite or null")
        elif kind == "categorical":
            if not isinstance(value, str):
                raise TypeError(f"Feature {feature!r} must be a string or null")
        else:
            raise ValueError(f"Missing or unsupported type declaration for {feature!r}")
        validated[feature] = value
    return pd.DataFrame([validated], columns=features)


def predict_one(artifact: dict[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a probability and non-causal explanation; no default decision threshold."""
    schema = artifact["feature_schema"]
    features = schema["features"]
    feature_types = schema["feature_types"]
    frame = validate_record(record, features, feature_types)
    predictions = artifact["model"].predict_proba(frame)
    if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] != 2:
        raise RuntimeError("Fitted model returned an unexpected probability shape")
    probability = float(predictions[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Fitted model returned an invalid probability")

    threshold = artifact.get("decision_threshold")
    decision = None
    if threshold is not None and artifact.get("decision_use_approved") is True:
        decision = bool(probability >= float(threshold))
    return {
        "model_version": artifact["model_version"],
        "target_status": artifact["target_status"],
        "probability": probability,
        "decision_threshold": threshold,
        "decision": decision,
        "explanation": {
            "scope": artifact.get("explanation_scope", "global"),
            "method": artifact.get("explanation_method", "not_available"),
            "top_inputs": artifact.get("global_importance", [])[:5],
            "note": "Associations are not causal explanations; the target is unverified.",
        },
    }
