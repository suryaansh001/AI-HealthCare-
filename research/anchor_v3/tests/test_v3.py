"""Unit tests for causal-order personal-history features and v3 diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.anchor_v3.features import (
    EMA_FEATURES,
    LONG_LAPSE_COUNT,
    RECENT_LAPSE_COUNT,
    add_personal_history_features,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for person in ("a", "b"):
        for index in range(12):
            row = {
                "subid": person,
                "dttm_label": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=index),
                "y": int(index % 2),
                RECENT_LAPSE_COUNT: float(index % 3),
                LONG_LAPSE_COUNT: float(index % 5),
            }
            row.update({feature: float(index + offset) for offset, feature in enumerate(EMA_FEATURES)})
            rows.append(row)
    return pd.DataFrame(rows)


class PersonalHistoryFeatureTests(unittest.TestCase):
    def test_features_use_only_strictly_prior_rows(self) -> None:
        original = sample_frame()
        transformed, families = add_personal_history_features(original)
        feature = "v3_ema_2_prior24_mean"
        first_person_rows = transformed.loc[transformed["subid"] == "a", feature].to_numpy()
        self.assertTrue(np.isnan(first_person_rows[:6]).all())
        self.assertEqual(first_person_rows[6], np.mean(np.arange(6)))
        self.assertEqual(
            families["personalized_hybrid"],
            families["personal_ema_baseline"] + families["history_rate_contrast"],
        )

        changed = original.copy()
        changed.loc[changed["subid"] == "a", EMA_FEATURES] = 9000.0
        changed_transformed, _ = add_personal_history_features(changed)
        first_person_changed = changed_transformed.loc[
            changed_transformed["subid"] == "a", feature
        ].to_numpy()
        self.assertTrue(np.isnan(first_person_changed[:6]).all())
        self.assertTrue(np.all(first_person_changed[6:] == 9000.0))

    def test_features_do_not_depend_on_target_values(self) -> None:
        original = sample_frame()
        baseline, _ = add_personal_history_features(original)
        changed = original.copy()
        changed["y"] = 1 - changed["y"]
        changed_features, _ = add_personal_history_features(changed)
        engineered = [column for column in baseline if column.startswith("v3_")]
        pd.testing.assert_frame_equal(
            baseline[engineered],
            changed_features[engineered],
        )

    def test_unsorted_rows_are_rejected(self) -> None:
        original = sample_frame().iloc[[1, 0] + list(range(2, 24))].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "sorted"):
            add_personal_history_features(original)


if __name__ == "__main__":
    unittest.main()
