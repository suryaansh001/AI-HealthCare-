"""Load and cross-check the preserved ANCHOR v2 feature contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT / "research" / "models" / "feature_schema_v2.json"
DEFAULT_CATEGORY_SCHEMA_PATH = (
    ROOT / "research" / "anchor_handover" / "feature_schema.json"
)


@dataclass(frozen=True)
class FeatureSchema:
    model_version: str
    features: tuple[str, ...]
    feature_types: dict[str, str]
    categorical_values: dict[str, tuple[str, ...]]
    semantics_status: str
    source_path: Path


def load_schema(
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    category_schema_path: str | Path = DEFAULT_CATEGORY_SCHEMA_PATH,
) -> FeatureSchema:
    """Load v2 names/types and the fitted category vocabulary used by its model."""
    schema_file = Path(schema_path)
    category_file = Path(category_schema_path)
    payload: dict[str, Any] = json.loads(schema_file.read_text(encoding="utf-8"))
    categories_payload: dict[str, Any] = json.loads(
        category_file.read_text(encoding="utf-8")
    )
    features = payload.get("features")
    feature_types = payload.get("feature_types")
    categories = categories_payload.get("categorical_values")
    if (
        not isinstance(features, list)
        or len(features) != 22
        or len(features) != len(set(features))
        or payload.get("feature_count") != len(features)
        or not isinstance(feature_types, dict)
        or set(feature_types) != set(features)
        or any(kind not in {"numeric", "categorical"} for kind in feature_types.values())
    ):
        raise ValueError(f"Invalid v2 model schema: {schema_file}")
    if (
        categories_payload.get("features") != features
        or categories_payload.get("feature_types") != feature_types
        or not isinstance(categories, dict)
        or set(categories)
        != {feature for feature, kind in feature_types.items() if kind == "categorical"}
        or any(
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            for values in categories.values()
        )
    ):
        raise ValueError(
            "Handover category vocabulary does not match the v2 feature schema"
        )
    return FeatureSchema(
        model_version=str(payload["model_version"]),
        features=tuple(features),
        feature_types=dict(feature_types),
        categorical_values={
            name: tuple(values) for name, values in categories.items()
        },
        semantics_status=str(payload.get("feature_semantics_status", "UNRESOLVED")),
        source_path=schema_file.resolve(),
    )
