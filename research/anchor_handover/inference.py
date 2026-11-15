"""Strict handover entrypoint for the preserved ANCHOR v2 artifact."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

from research.anchor_v2.inference import load_artifact as _load_artifact
from research.anchor_v2.inference import predict_one as _predict_one


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "research" / "models" / "anchor_lapse_model_v2.joblib"
SCHEMA_PATH = PACKAGE_DIR / "feature_schema.json"
SCHEMA_FIELDS = (
    "model_version",
    "feature_count",
    "features",
    "feature_types",
    "feature_semantics_status",
)


def _read_schema() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema.get("features")
    feature_types = schema.get("feature_types")
    categorical_values = schema.get("categorical_values")
    if (
        not isinstance(features, list)
        or len(features) != 22
        or len(features) != len(set(features))
        or not isinstance(feature_types, dict)
        or set(feature_types) != set(features)
        or not isinstance(categorical_values, dict)
        or set(categorical_values)
        != {name for name, kind in feature_types.items() if kind == "categorical"}
    ):
        raise ValueError("Handover feature schema is invalid")
    return schema


def _fitted_categories(artifact: dict[str, Any]) -> dict[str, list[str]]:
    try:
        preprocessor = artifact["model"].named_steps["preprocessor"]
        categorical_features = preprocessor.transformers_[0][2]
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    except (AttributeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("Saved model does not match the documented preprocessing pipeline") from error
    return {
        feature: [str(value) for value in categories]
        for feature, categories in zip(categorical_features, encoder.categories_)
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and cross-check the preserved model, schema, and inference policy."""
    artifact = _load_artifact(model_path)
    schema = _read_schema()
    embedded_schema = artifact["feature_schema"]
    if any(embedded_schema.get(key) != schema.get(key) for key in SCHEMA_FIELDS):
        raise ValueError("Handover schema does not match the model's embedded feature schema")
    if artifact["model_version"] != schema["model_version"]:
        raise ValueError("Model version does not match the handover schema")
    preprocessor = artifact["model"].named_steps.get("preprocessor")
    model_features = getattr(preprocessor, "feature_names_in_", None)
    if model_features is None or list(model_features) != schema["features"]:
        raise ValueError("Saved model preprocessing feature order differs from the schema")
    if _fitted_categories(artifact) != schema["categorical_values"]:
        raise ValueError("Categorical vocabulary differs from the fitted model")
    if artifact.get("decision_threshold") is not None or artifact.get("decision_use_approved") is not False:
        raise ValueError("A decision threshold is not approved for this unverified target")
    if artifact["target_status"] != "NOT_VERIFIED":
        raise ValueError("Handover requires the model target status NOT_VERIFIED")
    artifact["handover_schema"] = schema
    return artifact


def validate_input(record: Mapping[str, Any], artifact: dict[str, Any]) -> None:
    """Reject incomplete, extra, mistyped, non-finite, or unknown-coded inputs."""
    if not isinstance(record, Mapping):
        raise TypeError("Inference input must be a mapping of feature names to values")
    if any(not isinstance(key, str) for key in record):
        raise TypeError("All input feature names must be strings")

    schema = artifact["handover_schema"]
    features = schema["features"]
    expected = set(features)
    actual = set(record)
    missing = [feature for feature in features if feature not in actual]
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Input schema mismatch; missing={missing}, extra={extra}")

    for feature in features:
        value = record[feature]
        if value is None:
            continue
        kind = schema["feature_types"][feature]
        if kind == "numeric":
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"Feature {feature!r} must be a finite number or null")
            if not math.isfinite(float(value)):
                raise ValueError(f"Feature {feature!r} must be finite or null")
        elif kind == "categorical":
            if not isinstance(value, str):
                raise TypeError(f"Feature {feature!r} must be a string or null")
            if value not in schema["categorical_values"][feature]:
                raise ValueError(
                    f"Feature {feature!r} has an unknown category; "
                    f"expected one of {schema['categorical_values'][feature]}"
                )
        else:
            raise ValueError(f"Unsupported feature type for {feature!r}")


def predict_one(artifact: dict[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a probability and global explanation without a decision or threshold."""
    validate_input(record, artifact)
    result = _predict_one(artifact, record)
    probability = result.get("probability")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, numbers.Real)
        or not math.isfinite(float(probability))
        or not 0 <= float(probability) <= 1
    ):
        raise RuntimeError("Model returned an invalid probability")
    result["decision_threshold"] = None
    result["decision"] = None
    return result
