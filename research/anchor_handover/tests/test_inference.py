"""Integration tests against the preserved v2 model artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from research.anchor_handover.inference import (
    DEFAULT_MODEL_PATH,
    load_model,
    predict_one,
    validate_input,
)
from research.anchor_handover.predict import main as predict_main


class HandoverInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_model()
        cls.schema = cls.artifact["handover_schema"]
        cls.record = {
            feature: (
                cls.schema["categorical_values"][feature][0]
                if cls.schema["feature_types"][feature] == "categorical"
                else 0.0
            )
            for feature in cls.schema["features"]
        }

    def test_pinned_artifact_and_feature_order_load(self) -> None:
        self.assertTrue(DEFAULT_MODEL_PATH.is_file())
        self.assertEqual(self.artifact["model_version"], "2.0.0")
        self.assertEqual(self.artifact["target_status"], "NOT_VERIFIED")
        preprocessor = self.artifact["model"].named_steps["preprocessor"]
        self.assertEqual(
            list(preprocessor.feature_names_in_),
            self.schema["features"],
        )
        self.assertEqual(len(self.schema["features"]), 22)
        self.assertIsNone(self.artifact["decision_threshold"])
        self.assertFalse(self.artifact["decision_use_approved"])

    def test_repeated_prediction_is_finite_stable_and_has_no_decision(self) -> None:
        first = predict_one(self.artifact, self.record)
        second = predict_one(self.artifact, dict(reversed(list(self.record.items()))))
        self.assertAlmostEqual(first["probability"], second["probability"], places=12)
        self.assertGreaterEqual(first["probability"], 0.0)
        self.assertLessEqual(first["probability"], 1.0)
        self.assertEqual(first["model_version"], "2.0.0")
        self.assertEqual(first["target_status"], "NOT_VERIFIED")
        self.assertIsNone(first["decision"])
        self.assertIsNone(first["decision_threshold"])
        self.assertEqual(first["explanation"]["method"], "tree impurity importance")

    def test_explicit_nulls_use_saved_preprocessing(self) -> None:
        record = {feature: None for feature in self.schema["features"]}
        result = predict_one(self.artifact, record)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_missing_extra_and_wearable_fields_are_rejected(self) -> None:
        missing = dict(self.record)
        missing.pop(self.schema["features"][0])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_input(missing, self.artifact)

        extra = {**self.record, "unmodeled_wearable_hr": 72.0}
        with self.assertRaisesRegex(ValueError, "extra"):
            validate_input(extra, self.artifact)

    def test_invalid_numeric_values_are_rejected(self) -> None:
        numeric = next(
            feature
            for feature in self.schema["features"]
            if self.schema["feature_types"][feature] == "numeric"
        )
        for value in (True, "1.0"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_input({**self.record, numeric: value}, self.artifact)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_input({**self.record, numeric: value}, self.artifact)

    def test_invalid_categorical_values_are_rejected(self) -> None:
        categorical = next(
            feature
            for feature in self.schema["features"]
            if self.schema["feature_types"][feature] == "categorical"
        )
        with self.assertRaises(TypeError):
            validate_input({**self.record, categorical: 1}, self.artifact)
        with self.assertRaisesRegex(ValueError, "unknown category"):
            validate_input({**self.record, categorical: "__unknown__"}, self.artifact)

    def test_cli_rejects_nonstandard_nan_json(self) -> None:
        record = dict(self.record)
        numeric = next(
            feature
            for feature in self.schema["features"]
            if self.schema["feature_types"][feature] == "numeric"
        )
        record[numeric] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-standard JSON"):
                with redirect_stdout(StringIO()):
                    predict_main(["--model", str(DEFAULT_MODEL_PATH), "--input", str(input_path)])

    def test_cli_emits_contract_output_for_valid_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(json.dumps(self.record), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                predict_main(["--model", str(DEFAULT_MODEL_PATH), "--input", str(input_path)])
        result = json.loads(output.getvalue())
        self.assertEqual(result["model_version"], "2.0.0")
        self.assertEqual(result["target_status"], "NOT_VERIFIED")
        self.assertIsNone(result["decision"])
        self.assertIsNone(result["decision_threshold"])


if __name__ == "__main__":
    unittest.main()
