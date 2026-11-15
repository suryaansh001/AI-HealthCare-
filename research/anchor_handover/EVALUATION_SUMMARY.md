# ANCHOR model handover evaluation summary

## Selection

**Use v2 only as the technical reference model; v3 remains experimental.**
The preserved v2 artifact has a complete fitted preprocessing pipeline and
tested input validation. The v3 EMA personal-history hybrid has higher
exploratory stored-label metrics, but has no serialized artifact or tested
inference contract and uses features whose availability at prediction time is
unverified. Do not promote v3 based on the locked-test comparison.

## Stored-label evaluation

The locked v2 test contains 6,068 rows from 20 unseen participants (485
positive stored labels). V2 results were AUROC 0.8431, AUPRC 0.3522, Brier
0.06099, log loss 0.21417, and within-participant AUROC 0.4855 among 12
participants eligible for that metric.

The v3 EMA personal-history hybrid scored AUROC 0.8710, AUPRC 0.3877, Brier
0.05943, log loss 0.20331, and within-participant AUROC 0.5242 on the same
stored-label test rows. It produced 25.9 false alerts per 100 rows at its
exploratory threshold, versus 19.2 for v2. These exploratory metrics do not
establish prospective prediction, clinical validity, or reliable
personalization. Details are in
`../anchor_v3/ANCHOR_V3_PERSONALIZATION_RESULTS.md`.

## Inference behavior

The v2 artifact reloads with Python 3.10.19, NumPy 1.26.4, pandas 2.3.3,
SciPy 1.15.3, scikit-learn 1.7.2, and joblib 1.5.3. It contains a
100-tree Random Forest pipeline with numeric median imputation and categorical
constant imputation plus one-hot encoding. The model is uncalibrated.

The handover inference entrypoint validates the exact 22 names, schema types,
categorical vocabulary, finite numeric values, and null policy. It returns a
probability, model version, target status, global feature-importance
explanation, and null decision fields. Repeated predictions were stable to
12 decimal places. This verifies technical reload/inference behavior only.

## Recommendation and limitation

Retain v2 as the only packaged model artifact. The output must remain a
research score with no alert tier or decision. The target's exact temporal
meaning and feature timing remain **NOT VERIFIED**; all performance describes
agreement with stored labels, not future alcohol-lapse prediction. V2's
within-participant ranking is near chance, and no clinical utility is
established.
