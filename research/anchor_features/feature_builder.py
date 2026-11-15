"""Build exact model inputs from the preserved engineered v2 feature matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.anchor_features.schema import (
    DEFAULT_SCHEMA_PATH,
    ROOT,
    FeatureSchema,
    load_schema,
)
from research.anchor_features.validation import (
    IDENTITY_COLUMNS,
    validate_feature_mapping,
    validate_model_ready_frame,
    validate_raw_ema_frame,
    require_verified_raw_feature_generation,
    normalize_identity,
)
from research.anchor_handover.inference import load_model, predict_one


DEFAULT_SOURCE = ROOT / "dataset" / "completedatasetwithALLcsv" / "features_24lag_v3.csv"
DEFAULT_OUTPUT_DIR = ROOT / "research" / "anchor_features" / "generated"


@dataclass(frozen=True)
class FeatureBuildResult:
    """Model input frame, aligned row metadata, and source-provenance record."""

    features: pd.DataFrame
    metadata: pd.DataFrame
    manifest: dict[str, Any]


def build_from_engineered_matrix(
    source: pd.DataFrame,
    schema: FeatureSchema | None = None,
) -> FeatureBuildResult:
    """Select and validate existing engineered values; do not recompute transforms."""
    active_schema = schema or load_schema()
    required = set(IDENTITY_COLUMNS).union(active_schema.features)
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Engineered source matrix is missing required columns: {missing}")

    source = source.reset_index(drop=True)
    metadata = normalize_identity(source, *IDENTITY_COLUMNS)
    metadata = metadata.assign(source_row=range(len(source)))
    order = (
        metadata.assign(_position=range(len(metadata)))
        .sort_values(["subid", "dttm_label", "_position"], kind="stable")["_position"]
        .to_numpy()
    )
    source_rows = source.iloc[order].reset_index(drop=True)
    metadata = metadata.iloc[order].reset_index(drop=True)
    features = validate_model_ready_frame(
        source_rows.loc[:, list(active_schema.features)],
        active_schema,
    )
    return FeatureBuildResult(
        features=features.reset_index(drop=True),
        metadata=metadata,
        manifest={
            "model_version": active_schema.model_version,
            "schema_path": str(active_schema.source_path),
            "input_kind": "existing_engineered_feature_matrix",
            "source_rows": int(len(source)),
            "output_rows": int(len(features)),
            "feature_count": int(len(active_schema.features)),
            "feature_order": list(active_schema.features),
            "target_or_outcome_columns_read": False,
            "raw_to_feature_transforms_recomputed": False,
            "feature_timing_status": "NOT_VERIFIED",
            "historical_feature_generation_status": "UNRESOLVED; values copied from source matrix",
            "ema_feature_generation_status": "UNRESOLVED; values copied from source matrix",
            "demographic_and_label_day_generation_status": (
                "UNRESOLVED; values copied from source matrix"
            ),
        },
    )


def build_from_raw_ema(
    source: pd.DataFrame,
    schema: FeatureSchema | None = None,
) -> FeatureBuildResult:
    """Validate raw EMA structure, then fail rather than guess feature transforms."""
    active_schema = schema or load_schema()
    validate_raw_ema_frame(source)
    require_verified_raw_feature_generation(active_schema.features)


def generate_from_csv(
    source_path: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    schema: FeatureSchema | None = None,
    validate_inference: bool = True,
    limit: int | None = None,
) -> FeatureBuildResult:
    """Write an exact 22-column matrix plus aligned metadata and a provenance manifest."""
    active_schema = schema or load_schema()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {destination}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    columns = [*IDENTITY_COLUMNS, *active_schema.features]
    source = pd.read_csv(source_path, usecols=columns, low_memory=False, nrows=limit)
    result = build_from_engineered_matrix(source, active_schema)
    if validate_inference and not result.features.empty:
        artifact = load_model()
        first_record = validate_feature_mapping(
            result.features.iloc[0].to_dict(),
            active_schema,
        )
        predict_one(artifact, first_record)
        result.manifest["v2_inference_smoke_test"] = "passed on first sorted output row"
    else:
        result.manifest["v2_inference_smoke_test"] = "not run"

    destination.mkdir(parents=True)
    result.features.to_csv(destination / "model_features.csv", index=False)
    result.metadata.to_csv(destination / "row_metadata.csv", index=False)
    (destination / "manifest.json").write_text(
        json.dumps(result.manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-inference-smoke-test", action="store_true")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    arguments = parser.parse_args(argv)
    result = generate_from_csv(
        arguments.input,
        arguments.output_dir,
        schema=load_schema(arguments.schema),
        validate_inference=not arguments.skip_inference_smoke_test,
        limit=arguments.limit,
    )
    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir.resolve()),
                "rows": len(result.features),
                "features": len(result.features.columns),
                "manifest": result.manifest,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
