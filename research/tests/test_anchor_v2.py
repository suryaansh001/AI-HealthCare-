"""Focused tests for ANCHOR v2 validation and inference contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research.anchor_v2.data import encode_binary_target, load_feature_matrix
from research.anchor_v2.inference import predict_one, validate_record
from research.anchor_v2.pipeline import make_model_pipeline
from research.anchor_v2.predict import main as predict_main
from research.anchor_v2.validation import (
    TargetTimingSpec,
    TargetTimingUnverified,
    grouped_partitions,
    make_purged_chronological_split,
)


class DataValidationTests(unittest.TestCase):
    def test_binary_target_is_strict(self) -> None:
        encoded = encode_binary_target(pd.Series(["yes", "No", "1", "false"]))
        self.assertEqual(encoded.tolist(), [1, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, "Unknown or missing"):
            encode_binary_target(pd.Series(["yes", None]))

    def test_loader_checks_duplicate_timestamps_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema_path = root / "schema.json"
            data_path = root / "data.csv"
            schema_path.write_text('{"features":["x","category"]}', encoding="utf-8")
            pd.DataFrame(
                {
                    "subid": ["b", "a"],
                    "dttm_label": ["2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z"],
                    "lapse": ["no", "yes"],
                    "x": [None, 1.0],
                    "category": ["z", "a"],
                }
            ).to_csv(data_path, index=False)
            frame, _ = load_feature_matrix(data_path, schema_path, stride=1)
            self.assertEqual(frame["subid"].tolist(), ["a", "b"])
            self.assertEqual(frame["y"].tolist(), [1, 0])
            self.assertEqual(str(frame["dttm_label"].dtype), "datetime64[ns, UTC]")

            duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            duplicate.to_csv(data_path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_feature_matrix(data_path, schema_path, stride=1)


class ValidationTests(unittest.TestCase):
    def test_participant_groups_never_cross_partitions(self) -> None:
        groups = np.repeat(np.arange(70), 8)
        y = np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1]), 70)
        partitions = grouped_partitions(y, groups, n_splits=7, development_folds=3)
        group_sets = {name: set(groups[index]) for name, index in partitions.items()}
        for index, left in enumerate(group_sets.values()):
            for right in list(group_sets.values())[index + 1 :]:
                self.assertFalse(left.intersection(right))

    def test_chronological_split_requires_verified_target_and_purges_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "prediction": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-04",
                    ],
                    utc=True,
                ),
                "start": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-04",
                    ],
                    utc=True,
                ),
                "end": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-04",
                        "2020-01-04",
                        "2020-01-05",
                    ],
                    utc=True,
                ),
                "followup": pd.to_datetime(
                    [
                        "2020-01-05",
                        "2020-01-05",
                        "2020-01-05",
                        "2020-01-05",
                        None,
                    ],
                    utc=True,
                ),
            }
        )
        unverified = TargetTimingSpec(False, "prediction", "start", "end", "followup", "closed")
        with self.assertRaises(TargetTimingUnverified):
            make_purged_chronological_split(frame, "2020-01-03", unverified)

        verified = TargetTimingSpec(True, "prediction", "start", "end", "followup", "closed")
        split = make_purged_chronological_split(frame, "2020-01-03", verified)
        self.assertEqual(split.train_indices.tolist(), [0])
        self.assertEqual(split.test_indices.tolist(), [3])
        self.assertEqual(split.purged_training_rows, 2)
        self.assertEqual(split.incomplete_test_rows, 1)


class InferenceContractTests(unittest.TestCase):
    def test_schema_validation_and_reload_prediction(self) -> None:
        features = ["numeric", "category"]
        types = {"numeric": "numeric", "category": "categorical"}
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_record({"numeric": 1.0}, features, types)
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            validate_record({"numeric": "1", "category": "x"}, features, types)

        training = pd.DataFrame(
            {"numeric": [0.0, 1.0, 2.0, 3.0], "category": ["a", "a", "b", "b"]}
        )
        model = make_model_pipeline(
            training,
            features,
            "logistic",
            {"C": 1.0},
        ).fit(training, np.array([0, 0, 1, 1]))
        artifact = {
            "model_version": "test",
            "model": model,
            "feature_schema": {"features": features, "feature_types": types},
            "target_status": "NOT_VERIFIED",
            "decision_threshold": None,
            "decision_use_approved": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "model.joblib"
            input_path = Path(directory) / "input.json"
            joblib.dump(artifact, artifact_path)
            restored = joblib.load(artifact_path)
            result = predict_one(restored, {"numeric": 2.5, "category": "b"})
            again = predict_one(restored, {"numeric": 2.5, "category": "b"})
            input_path.write_text(
                '{"numeric":2.5,"category":"b"}',
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                predict_main(["--model", str(artifact_path), "--input", str(input_path)])
        self.assertEqual(result["probability"], again["probability"])
        self.assertIsNone(result["decision"])
        self.assertEqual(result["target_status"], "NOT_VERIFIED")
        self.assertEqual(
            json.loads(output.getvalue())["probability"],
            result["probability"],
        )


if __name__ == "__main__":
    unittest.main()
