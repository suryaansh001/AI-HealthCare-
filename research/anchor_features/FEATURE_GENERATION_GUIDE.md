# ANCHOR v2 Feature Preparation Guide

## Important scope

This package safely assembles model-ready rows from the existing
`dataset/completedatasetwithALLcsv/features_24lag_v3.csv` matrix. It selects and
validates the 22 precomputed model values, preserves participant/timestamp
metadata separately, sorts chronologically within participant, and checks one
output row through the v2 inference utility.

It does **not** reproduce the upstream feature calculations from raw EMA and
event data. Raw-to-feature definitions for the historical counts, the exact
EMA response transformations/lag alignment, demographics, and `label_day`
could not be verified in this checkout. `dataset/ema.csv` contains raw EMA
observations, but it does not supply the authoritative event table or the
join/aggregation rules needed to recreate the model's `l24` features.
Historical `lapse` labels are never read as predictors by this builder. It
does not infer distinct drinking events from affirmative EMA item responses.
For raw EMA-only input the API validates the raw structure and then fails
closed.

Thus this is a **model-input preparation adapter**, not a scientifically
verified raw-data feature generator. Matching the 22-column schema establishes
technical compatibility only; prediction-time availability remains
`NOT_VERIFIED`.

## Generate a model-ready file

From the repository root, using the tested project Python environment:

```powershell
python -m research.anchor_features.feature_builder `
  --input dataset\completedatasetwithALLcsv\features_24lag_v3.csv `
  --output-dir research\anchor_features\generated
```

The output directory must not already exist. The command writes:

- `model_features.csv` — exactly the 22 v2 features, in training order.
- `row_metadata.csv` — `subid`, UTC-normalized `dttm_label`, and original
  source-row index. Keep this file internal; it is not model input.
- `manifest.json` — schema version, row counts, source status, unresolved
  provenance, and v2 inference smoke-test result.

The CLI only reads the identity columns and the 22 schema feature columns. It
does not load `lapse`, `y`, or other outcome columns. Use `--limit 10` for a
small output/inference smoke run; select a new, non-existing output directory
for every run. `--skip-inference-smoke-test` disables the one-row smoke check
but is not recommended.

The API is `build_from_engineered_matrix(source, schema=None)`. The raw EMA
validator is `validate_raw_ema_frame(frame)`. Calling
`build_from_raw_ema(frame)` raises `UnresolvedFeatureSourceError` until the
authoritative source and transformation definitions are recovered.

## Schema and feature provenance

The names and types below come directly from
`research/models/feature_schema_v2.json`. Values are pass-through from the
existing engineered matrix; no transformations are invented here. Fitted
categorical vocabularies are cross-checked against the machine-readable
handover schema and saved v2 estimator.

| Feature(s) | Schema type | Existing source | Raw transformation / missingness / history status | Prediction-time status |
|---|---|---|---|---|
| `label_day` | categorical | Existing 24-lag matrix column | Category values are checked against fitted model; timezone/day construction is not reconstructed. Null is preserved for model imputation. | UNRESOLVED |
| `demo_age` | numeric | Existing 24-lag matrix column | Numeric finite value or null; no raw demographic source/mapping identified. | UNRESOLVED |
| `demo_sex` | categorical | Existing 24-lag matrix column | Fitted values: `Female`, `Male`; no recoding is performed. | UNRESOLVED |
| `demo_educ` | categorical | Existing 24-lag matrix column | Fitted values: `College or more`, `High school or less`, `Some college`; no recoding is performed. | UNRESOLVED |
| `demo_income` | numeric | Existing 24-lag matrix column | Numeric finite value or null; units/encoding not verified. | UNRESOLVED |
| `demo_marital` | categorical | Existing 24-lag matrix column | Fitted values: `Married`, `Never Married`, `Other`; no recoding is performed. | UNRESOLVED |
| `demo_race` | categorical | Existing 24-lag matrix column | Fitted values: `Other`, `White/Caucasian`; no recoding is performed. | UNRESOLVED |
| `demo_employ` | categorical | Existing 24-lag matrix column | Fitted values: `Employed`, `Unemployed`; no recoding is performed. | UNRESOLVED |
| `p12.l24.rratecount.count.lapse` | numeric | Existing 24-lag matrix column | Source dictionary labels a 12 h `past_use` raw `ratecount` of `lapse`; event definition, boundary, lag origin, and missingness rule are unavailable. | UNRESOLVED |
| `p12.l24.dratecount.count.lapse` | numeric | Existing 24-lag matrix column | Dictionary labels a 12 h change feature; exact change formula and required prior interval unavailable. | UNRESOLVED |
| `p24.l24.rratecount.count.lapse` | numeric | Existing 24-lag matrix column | Source dictionary labels a 24 h `past_use` raw `ratecount` of `lapse`; event definition, boundary, lag origin, and missingness rule are unavailable. | UNRESOLVED |
| `p24.l24.dratecount.count.lapse` | numeric | Existing 24-lag matrix column | Dictionary labels a 24 h change feature; exact change formula and required prior interval unavailable. | UNRESOLVED |
| `p48.l24.rratecount.count.lapse` | numeric | Existing 24-lag matrix column | Source dictionary labels a 48 h `past_use` raw `ratecount` of `lapse`; event definition, boundary, lag origin, and missingness rule are unavailable. | UNRESOLVED |
| `p48.l24.dratecount.count.lapse` | numeric | Existing 24-lag matrix column | Dictionary labels a 48 h change feature; exact change formula and required prior interval unavailable. | UNRESOLVED |
| `p72.l24.rratecount.count.lapse` | numeric | Existing 24-lag matrix column | Source dictionary labels a 72 h `past_use` raw `ratecount` of `lapse`; event definition, boundary, lag origin, and missingness rule are unavailable. | UNRESOLVED |
| `p72.l24.dratecount.count.lapse` | numeric | Existing 24-lag matrix column | Dictionary labels a 72 h change feature; exact change formula and required prior interval unavailable. | UNRESOLVED |
| `p168.l24.rratecount.count.lapse` | numeric | Existing 24-lag matrix column | Source dictionary labels a 168 h `past_use` raw `ratecount` of `lapse`; event definition, boundary, lag origin, and missingness rule are unavailable. No 168 h calculation is reconstructed. | UNRESOLVED |
| `p168.l24.dratecount.count.lapse` | numeric | Existing 24-lag matrix column | Dictionary labels a 168 h change feature; exact change formula and required prior interval unavailable. | UNRESOLVED |
| `ema_2.p0.l24.rrecent_response` | numeric | Existing 24-lag matrix column; raw item `ema_2` exists in `dataset/ema.csv` | Direct response mapping, timezone handling, tie/duplicate rule, missingness and 24 h alignment are not verified; no recoding is performed. | UNRESOLVED |
| `ema_6.p0.l24.rrecent_response` | numeric | Existing 24-lag matrix column; raw item `ema_6` exists in `dataset/ema.csv` | Direct response mapping, timezone handling, tie/duplicate rule, missingness and 24 h alignment are not verified; no recoding is performed. | UNRESOLVED |
| `ema_7.p0.l24.rrecent_response` | numeric | Existing 24-lag matrix column; raw item `ema_7` exists in `dataset/ema.csv` | Direct response mapping, timezone handling, tie/duplicate rule, missingness and 24 h alignment are not verified; no recoding is performed. | UNRESOLVED |
| `ema_10.p0.l24.rrecent_response` | numeric | Existing 24-lag matrix column; raw item `ema_10` exists in `dataset/ema.csv` | Direct response mapping, morning-only availability, timezone handling, tie/duplicate rule, missingness and 24 h alignment are not verified; no recoding is performed. | UNRESOLVED |

`dataset/column_dictionary.csv` identifies period/kind/statistic metadata for
related `l0` features, but it does not implement the builder, establish exact
event intervals, or verify those calculations for the schema's `l24` features.
The audit report
`research/audit/ANCHOR_TARGET_PROVENANCE_INVESTIGATION.md` further explains
why an affirmative EMA report cannot be assumed to be the event stream used by
the historical features or target labels.

## Validation and fail-safe behavior

- Participants and timestamps must be present; timestamps are normalized to
  UTC and invalid/missing values rejected.
- Duplicate `(subid, dttm_label)` keys are rejected, not silently deduplicated.
- Rows are sorted by participant then timestamp; output feature rows and
  metadata retain the same order.
- Only the exact 22 schema features are emitted. Extra source matrix columns
  are ignored and never become predictors; model-ready validation rejects
  missing, reordered, or additional feature columns.
- Numeric columns are converted to `float64`, with nulls preserved and
  nonnumeric/infinite values rejected.
- Categorical columns are checked against fitted V2 categories; nulls are
  preserved and unknown categories rejected.
- No historical values are calculated from the stored `lapse` column. No raw
  EMA values are transformed or aligned without authoritative rules.
- The saved V2 inference preprocessing handles explicit nulls. No arbitrary
  zeros or mock wearable/physiological values are inserted.
- The inference smoke test checks only schema/runtime compatibility. It does
  not establish source feature validity, future availability, or predictive
  validity.

## Requirements and tests

No new package dependency is required beyond the existing v2 environment.
Tested with Python 3.10.19, pandas 2.3.3, NumPy 1.26.4, and scikit-learn
1.7.2.

```powershell
python -m unittest research.anchor_features.tests.test_feature_builder -v
python -m unittest research.tests.test_anchor_v2 research.anchor_v3.tests.test_v3 research.anchor_handover.tests.test_inference -v
```
