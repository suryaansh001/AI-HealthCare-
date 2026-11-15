"""Validation and compatibility tests for schema-safe feature preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.anchor_features.feature_builder import (
    build_from_engineered_matrix,
    build_from_raw_ema,
    generate_from_csv,
)
from research.anchor_features.schema import load_schema
from research.anchor_features.validation import (
    UnresolvedFeatureSourceError,
    validate_feature_mapping,
    validate_model_ready_frame,
    validate_raw_ema_frame,
)
from research.anchor_handover.inference import load_model, predict_one


def source_frame() -> pd.DataFrame:
    schema = load_schema()
    rows = []
    for participant, start_hour, participant_marker in (
        ("person-b", 20, 20),
        ("person-a", 10, 10),
    ):
        for offset in (2, 0, 1):
            row = {
                "subid": participant,
                "dttm_label": f"2020-01-01T{start_hour + offset:02d}:00:00-06:00",
                "lapse": "yes" if offset == 1 else "no",
                "y": int(offset == 1),
                "ignored_future_or_outcome": "not a predictor",
            }
            for name in schema.features:
                if schema.feature_types[name] == "categorical":
                    row[name] = schema.categorical_values[name][0]
                else:
                    row[name] = float(participant_marker + offset)
            rows.append(row)
    return pd.DataFrame(rows)


class FeatureSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema()

    def test_exact_22_feature_order_and_types(self) -> None:
        result = build_from_engineered_matrix(source_frame(), self.schema)
        self.assertEqual(list(result.features.columns), list(self.schema.features))
        self.assertEqual(len(result.features.columns), 22)
        self.assertEqual(
            list(result.metadata.columns),
            ["subid", "dttm_label", "source_row"],
        )
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(result.features[name])
                for name, kind in self.schema.feature_types.items()
                if kind == "numeric"
            )
        )
        self.assertTrue(
            all(
                pd.api.types.is_string_dtype(result.features[name])
                for name, kind in self.schema.feature_types.items()
                if kind == "categorical"
            )
        )

    def test_rows_are_sorted_by_participant_and_utc_timestamp(self) -> None:
        result = build_from_engineered_matrix(source_frame(), self.schema)
        self.assertEqual(
            result.metadata["subid"].tolist(),
            ["person-a"] * 3 + ["person-b"] * 3,
        )
        for _, rows in result.metadata.groupby("subid", sort=False):
            self.assertTrue(rows["dttm_label"].is_monotonic_increasing)
            self.assertEqual(str(rows["dttm_label"].dtype), "datetime64[ns, UTC]")

    def test_duplicate_participant_timestamp_is_rejected(self) -> None:
        source = source_frame()
        duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_from_engineered_matrix(duplicate, self.schema)

    def test_missing_identity_or_feature_columns_are_rejected(self) -> None:
        source = source_frame().drop(columns=["subid"])
        with self.assertRaisesRegex(ValueError, "required columns"):
            build_from_engineered_matrix(source, self.schema)
        source = source_frame().drop(columns=[self.schema.features[-1]])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_from_engineered_matrix(source, self.schema)

    def test_missing_numeric_values_are_preserved_for_model_imputation(self) -> None:
        source = source_frame()
        numeric = next(
            name for name, kind in self.schema.feature_types.items() if kind == "numeric"
        )
        source.loc[0, numeric] = np.nan
        result = build_from_engineered_matrix(source, self.schema)
        self.assertTrue(result.features[numeric].isna().any())

    def test_invalid_types_and_non_finite_numbers_are_rejected(self) -> None:
        numeric = next(
            name for name, kind in self.schema.feature_types.items() if kind == "numeric"
        )
        categorical = next(
            name
            for name, kind in self.schema.feature_types.items()
            if kind == "categorical"
        )
        with self.assertRaises(TypeError):
            validate_model_ready_frame(
                pd.DataFrame([{**{name: 0.0 for name in self.schema.features}, numeric: "nope"}]),
                self.schema,
            )
        bad_number = {name: 0.0 for name in self.schema.features}
        for name, kind in self.schema.feature_types.items():
            if kind == "categorical":
                bad_number[name] = self.schema.categorical_values[name][0]
        bad_number[numeric] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_model_ready_frame(pd.DataFrame([bad_number]), self.schema)
        bad_category = {**bad_number, numeric: 0.0, categorical: 17}
        with self.assertRaises(TypeError):
            validate_model_ready_frame(pd.DataFrame([bad_category]), self.schema)

    def test_missing_and_extra_model_columns_are_rejected(self) -> None:
        valid = source_frame().loc[:, list(self.schema.features)]
        with self.assertRaisesRegex(ValueError, "columns mismatch"):
            validate_model_ready_frame(valid.drop(columns=[self.schema.features[0]]), self.schema)
        with self.assertRaisesRegex(ValueError, "columns mismatch"):
            validate_model_ready_frame(
                valid.assign(extra_field=1),
                self.schema,
            )

    def test_unknown_category_is_rejected(self) -> None:
        record = {
            name: (
                self.schema.categorical_values[name][0]
                if kind == "categorical"
                else 0.0
            )
            for name, kind in self.schema.feature_types.items()
        }
        category = next(iter(self.schema.categorical_values))
        record[category] = "__unknown__"
        with self.assertRaisesRegex(ValueError, "unknown category"):
            validate_feature_mapping(record, self.schema)

    def test_participant_features_are_not_mixed_and_outcomes_are_excluded(self) -> None:
        source = source_frame()
        result = build_from_engineered_matrix(source, self.schema)
        self.assertNotIn("subid", result.features.columns)
        self.assertNotIn("lapse", result.features.columns)
        self.assertNotIn("y", result.features.columns)
        feature_name = self.schema.features[8]
        for participant, participant_marker in (
            ("person-a", 10.0),
            ("person-b", 20.0),
        ):
            selected = (
                result.metadata["subid"].eq(participant).to_numpy()
            )
            self.assertEqual(
                result.features.loc[selected, feature_name].tolist(),
                [
                    participant_marker,
                    participant_marker + 1,
                    participant_marker + 2,
                ],
            )

    def test_adding_future_rows_does_not_change_existing_output(self) -> None:
        source = source_frame()
        baseline = build_from_engineered_matrix(source, self.schema)
        future = source.iloc[[0]].copy()
        future["dttm_label"] = "2021-01-01T00:00:00Z"
        future["subid"] = "person-a"
        future["lapse"] = "yes"
        future["y"] = 1
        extended = build_from_engineered_matrix(
            pd.concat([source, future], ignore_index=True),
            self.schema,
        )
        previous = extended.metadata["dttm_label"] < pd.Timestamp(
            "2021-01-01T00:00:00Z"
        )
        pd.testing.assert_frame_equal(
            baseline.features.reset_index(drop=True),
            extended.features.loc[previous].reset_index(drop=True),
        )

    def test_feature_pass_through_is_repeatable_and_does_not_recompute_history(self) -> None:
        source = source_frame()
        first = build_from_engineered_matrix(source, self.schema)
        second = build_from_engineered_matrix(source, self.schema)
        pd.testing.assert_frame_equal(first.features, second.features)
        for feature in self.schema.features:
            if self.schema.feature_types[feature] == "numeric":
                expected = source.sort_values(["subid", "dttm_label"])[feature]
                np.testing.assert_allclose(
                    first.features[feature].to_numpy(),
                    expected.to_numpy(dtype=float),
                )
        self.assertFalse(first.manifest["raw_to_feature_transforms_recomputed"])

    def test_raw_ema_validation_sorts_and_rejects_bad_data(self) -> None:
        ema = pd.DataFrame(
            {
                "subid": ["b", "a"],
                "dttm_obs": ["2020-01-02T00:00:00Z", "2020-01-01T00:00:00-06:00"],
                "ema_2": [1, 2],
                "ema_6": [3, 4],
                "ema_7": [5, 6],
                "ema_10": [7, 8],
            }
        )
        normalized = validate_raw_ema_frame(ema)
        self.assertEqual(normalized["subid"].tolist(), ["a", "b"])
        bad = ema.copy()
        bad["ema_2"] = bad["ema_2"].astype(float)
        bad.loc[0, "ema_2"] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_raw_ema_frame(bad)
        duplicate = pd.concat([ema, ema.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_raw_ema_frame(duplicate)

    def test_raw_ema_cannot_be_guessed_into_model_features(self) -> None:
        ema = pd.DataFrame(
            {
                "subid": ["a"],
                "dttm_obs": ["2020-01-01T00:00:00Z"],
                "ema_2": [1.0],
                "ema_6": [2.0],
                "ema_7": [3.0],
                "ema_10": [4.0],
            }
        )
        with self.assertRaisesRegex(UnresolvedFeatureSourceError, "not verified"):
            build_from_raw_ema(ema, self.schema)

    def test_existing_matrix_output_passes_actual_v2_inference(self) -> None:
        source = source_frame()
        result = build_from_engineered_matrix(source, self.schema)
        artifact = load_model()
        record = validate_feature_mapping(result.features.iloc[0].to_dict(), self.schema)
        prediction = predict_one(artifact, record)
        self.assertEqual(prediction["model_version"], "2.0.0")
        self.assertEqual(prediction["target_status"], "NOT_VERIFIED")
        self.assertTrue(np.isfinite(prediction["probability"]))
        self.assertGreaterEqual(prediction["probability"], 0.0)
        self.assertLessEqual(prediction["probability"], 1.0)

    def test_csv_cli_writes_new_bundle_and_manifest(self) -> None:
        source = source_frame()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "matrix.csv"
            output_dir = root / "generated"
            source.to_csv(input_path, index=False)
            result = generate_from_csv(
                input_path,
                output_dir,
                schema=self.schema,
                validate_inference=True,
                limit=4,
            )
            features = pd.read_csv(output_dir / "model_features.csv")
            metadata = pd.read_csv(output_dir / "row_metadata.csv")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(list(features.columns), list(self.schema.features))
            self.assertEqual(len(features), len(metadata))
            self.assertEqual(len(result.features), 4)
            self.assertEqual(manifest["v2_inference_smoke_test"], "passed on first sorted output row")
            with self.assertRaises(FileExistsError):
                generate_from_csv(
                    input_path,
                    output_dir,
                    schema=self.schema,
                    validate_inference=False,
                    limit=1,
                )


if __name__ == "__main__":
    unittest.main()
