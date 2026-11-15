# ANCHOR ML handover package

This directory is the ML-to-backend handover. **Use v2 as the only technical
model artifact. V3 remains experimental and is not packaged for inference.**

## Backend developer: start here

1. Read [`BACKEND_CONTRACT.md`](./BACKEND_CONTRACT.md) before integrating.
   It contains the exact 22-field request schema, types, categorical values,
   missing-value behavior, output shape, and data-history caveats.
2. Use [`feature_schema.json`](./feature_schema.json) as the machine-readable
   input order and type schema.
3. Use `research.anchor_handover.inference.load_model()` and
   `research.anchor_handover.inference.predict_one()` (or the CLI below).
4. Load the preserved model from
   `research/models/anchor_lapse_model_v2.joblib`; do not copy the large model
   into this directory. Verify its digest in
   [`MODEL_ARTIFACT_REFERENCE.json`](./MODEL_ARTIFACT_REFERENCE.json).
5. Install the exact tested environment listed in [`requirements.txt`](./requirements.txt).

## Package contents

- `BACKEND_CONTRACT.md` — required input/output and integration policy.
- `feature_schema.json` — exact feature names/order/types and categorical vocabulary.
- `inference.py`, `predict.py` — strict handover validation and inference entrypoint.
- `MODEL_ARTIFACT_REFERENCE.json` — preserved artifact path, hash, and metadata.
- `model_card.json` — intended use and limitations.
- `EVALUATION_SUMMARY.md` — v2/v3 comparison and final model recommendation.
- `requirements.txt` — exact tested inference dependencies.
- `tests/` — integration tests against the preserved v2 model.

## Quick start

From the repository root:

```powershell
python -m research.anchor_handover.predict --input path\to\features.json
python -m unittest research.anchor_handover.tests.test_inference -v
```

Input must contain exactly 22 required fields. All may be explicit JSON `null`
and will be imputed by the saved preprocessing pipeline, but null is not
interchangeable with zero. Unknown categories, non-finite numbers, missing
fields, and extra fields are rejected.

The output contains an uncalibrated stored-label score with `target_status`
`NOT_VERIFIED`; decision and threshold are null. Do not use it to initiate
alerts or interventions. Exact target timing, feature timing, and clinical
validity remain unresolved.
