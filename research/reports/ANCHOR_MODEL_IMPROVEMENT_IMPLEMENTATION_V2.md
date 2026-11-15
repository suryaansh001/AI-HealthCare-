# ANCHOR Model Improvement Implementation v2

**Date:** 2026-11-15  
**Repository state:** `anchor-research-correction` at `1daa744959e02b94a3914e7ae7b30415aa8a0053`  
**Scope:** New research pipeline and artifacts only; no existing source, partner UI/backend, audit report, or model artifact was overwritten.

## Outcome and scientific boundary

A reproducible v2 training, grouped evaluation, calibration comparison, threshold analysis, versioned artifact, and inference CLI were implemented and run. The local `lapse` target's provenance and temporal meaning remain **NOT VERIFIED**. All metrics below describe agreement with the stored labels; they do not establish a future horizon, prospective warning, personalized forecasting, clinical validity, or utility.

The chronological split utility is implemented with strict verification and censoring/purging requirements, but the current matrix lacks verified outcome-start, outcome-end, and follow-up fields. It deliberately refuses to run on the current data. No target definition or future feature was invented.

## Implemented pipeline

* **Data checks:** strict binary target parsing (unknown/missing labels fail), participant and timestamp validation, duplicate participant/timestamp rejection, UTC parsing, stable sorting, and stride sampling after sorting. The 24-hour matrix has 270,911 source rows; stride 6 yields 45,206 rows from 151 participants. The compact model inputs had 437 missing cells, handled within model pipelines.
* **Preprocessing:** median imputation for numeric inputs, constant imputation and one-hot encoding for categoricals, and scaling for logistic regression. Every transform is fitted within the corresponding training fold.
* **Unseen-participant track:** a seven-way participant-grouped partition reserves 64 participants for development and separate participant groups for calibration fitting (22), calibration selection (22), threshold selection (23), and locked test (20). Candidate selection uses five-fold grouped OOF predictions within development only. No participant crosses any partition.
* **Model and ablation comparison:** 24 fixed candidate configurations spanning Logistic Regression, Random Forest, and HistGradientBoosting; the 22-input compact set plus historical lapse only, EMA only, demographics only, historical+EMA, and all available predictors. Small hyperparameter variants were restricted to the compact set.
* **Selection rule:** mean rank across development-only AUROC, AUPRC, Brier score, and log loss; smaller feature count breaks ties. The locked test was not used for candidate selection.
* **Calibration:** uncalibrated, sigmoid, and isotonic probability outputs were fitted/evaluated using participant-disjoint calibration groups. Isotonic was enabled only after the calibration-fit data met minimum class counts and participant-count criteria. The selected method was **uncalibrated**.
* **Threshold:** an exploratory Youden-J threshold was calculated on a separate participant group. It is not stored as an operational decision threshold and the inference utility returns no decision, because target semantics and alert consequences are not verified.
* **Chronological track:** `make_purged_chronological_split` requires a verified target specification plus explicit prediction, outcome-start, outcome-end, and follow-up timestamps. It excludes incomplete follow-up and training outcomes ending at or after the cutoff (a conservative boundary purge). The local data cannot currently satisfy that contract.
* **Features:** no new derived predictor was added. Existing features were grouped for controlled comparisons; data meaning and temporal availability remain unverified. No wearable/physiology inputs or simulated UI values were used.

## Results

### Selected candidate and held-out participants

The development-only grouped OOF comparison selected `random_forest__compact_22` (100 trees, max depth 8, minimum leaf size 50). Its development OOF metrics were AUROC **0.8856**, AUPRC **0.5109**, Brier **0.05940**, and log loss **0.20566**. These were used for selection, not as the final performance estimate.

The independently locked participant test contained 6,068 sampled rows and 20 participants (485 positive rows; prevalence 0.07993):

| Metric | Locked-test result |
|---|---:|
| AUROC | 0.8431 |
| AUPRC | 0.3522 |
| Brier score | 0.06099 |
| Log loss | 0.21417 |
| 95% participant-bootstrap AUROC interval | [0.7460, 0.8978] |
| 95% participant-bootstrap AUPRC interval | [0.1223, 0.5432] |
| 95% participant-bootstrap Brier interval | [0.0295, 0.0993] |

Within-participant AUROC was computable for 12 of the 20 test participants with at least three positive and three negative rows: macro AUROC **0.4855**, participant-bootstrap 95% interval **[0.3733, 0.5940]**. This is not a personalized chronological evaluation.

### Calibration comparison

| Probability method | Calibration-selection Brier | Calibration-selection log loss | Locked-test AUROC | Locked-test AUPRC | Locked-test Brier | Locked-test log loss |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.04181 | 0.14739 | 0.8431 | 0.3522 | 0.06099 | 0.21417 |
| Sigmoid | 0.04249 | 0.15541 | 0.8431 | 0.3522 | 0.06349 | 0.22421 |
| Isotonic | 0.04338 | 0.17870 | 0.8402 | 0.3194 | 0.06135 | 0.26285 |

The selection partition chose uncalibrated probabilities by the predeclared Brier/log-loss rank rule. No calibration is claimed.

### Separate threshold analysis

The exploratory Youden-J threshold selected on the threshold-only participant partition was **0.08858**. On the untouched test partition it gave sensitivity **0.7753**, specificity **0.7908**, precision **0.2435**, F1 **0.3706**, alert rate **25.44%**, and **19.25 false alerts per 100 sampled rows**. Rows are stride-six samples from an hourly matrix, so this is not an alert-per-day estimate. The threshold is not enabled for inference.

### Feature-family ablation (development OOF)

The following are the best-ranked candidate within each feature family on the same development folds; they are exploratory and not independent final-test comparisons.

| Feature family | Best candidate | Inputs | AUROC | AUPRC | Brier |
|---|---|---:|---:|---:|---:|
| Compact schema | Random Forest | 22 | 0.8856 | 0.5109 | 0.05940 |
| Historical lapse only | Logistic Regression | 10 | 0.8789 | 0.5125 | 0.06026 |
| EMA only | Random Forest | 252 | 0.8379 | 0.2889 | 0.07022 |
| Demographic only | Random Forest | 7 | 0.5235 | 0.1141 | 0.08871 |
| Historical + EMA | Random Forest | 262 | 0.8789 | 0.4229 | 0.06298 |
| Full available predictors | Random Forest | 280 | 0.8790 | 0.4266 | 0.06287 |

The history-only development result is not proof of target leakage or of valid history timing; the local outcome specification and feature-generation timestamps remain unavailable. Demographic meanings are also unverified, so no demographic subgroup claims are made.

### Context versus historical results

The previous all-participant grouped OOF result for `Logit_Calibrated_K` was AUROC **0.8713**, AUPRC **0.4748**, Brier **0.05365**, and within-person AUROC **0.5812**. The v2 locked-test result uses different participants, a different selected model, and a distinct train/calibration/test design. These values are **not a direct like-for-like improvement comparison**. Both remain conditional on the unverified local labels.

## Artifact, schema, and inference

* New serialized artifact: [`anchor_lapse_model_v2.joblib`](../models/anchor_lapse_model_v2.joblib)
* New exact input order and types: [`feature_schema_v2.json`](../models/feature_schema_v2.json)
* Research-only card: [`model_card_v2.json`](../models/model_card_v2.json)
* Inference API: [`inference.py`](../anchor_v2/inference.py)
* JSON CLI: [`predict.py`](../anchor_v2/predict.py)

The artifact embeds fold-fitted preprocessing and the selected uncalibrated estimator. It records `target_status: NOT_VERIFIED`, has no active decision threshold, and must only be loaded from a trusted source (joblib is a pickle-based format). Inference rejects missing/extra fields, incompatible types, non-finite numeric values, and invalid probability output. It returns a probability, model version, target status, and global non-causal importance summary. It does not silently add mock data.

Reproducible commands from the repository root, using the recorded environment:

```powershell
& 'C:\Users\Mohit sharma\anaconda3\envs\chatbot_env\python.exe' -m research.anchor_v2.train --seed 42 --stride 6 --bootstrap-replicates 500
& 'C:\Users\Mohit sharma\anaconda3\envs\chatbot_env\python.exe' -m research.anchor_v2.predict --model research\models\anchor_lapse_model_v2.joblib --input <path-to-exact-schema-json>
```

Training refuses to overwrite an existing v2 output. The tested runtime was CPython 3.10.19, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.3, and PyArrow 23.0.0; matching pins are in [`requirements.txt`](../anchor_v2/requirements.txt).

## Files and verification

**Created:** `research/anchor_v2/{__init__.py,data.py,pipeline.py,validation.py,train.py,inference.py,predict.py,requirements.txt}`, `research/tests/test_anchor_v2.py`, `research/models/{anchor_lapse_model_v2.joblib,feature_schema_v2.json,model_card_v2.json}`, `research/validation/{model_comparison_v2.json,oof_predictions_v2.parquet}`, and this report.

**Modified existing files:** none. The three existing audit reports and v1 model files remain unchanged. No frontend/backend file, dataset, source file already present, or Git history was modified; no package was installed and nothing was staged or committed.

**Verification run:**

* `python -m unittest discover -s research\tests -v` — **5 tests passed**.
* Pylance syntax checks and workspace diagnostics for new Python files — no syntax errors or reported problems.
* Full fixed-seed grouped evaluation — 24 candidate configurations completed.
* New artifact reload and inference smoke check on one schema-shaped dataset row — probability was in range and repeated reload calls matched. This is a technical check, not predictive validation.
* Chronological split unit test — verified synthetic intervals at and across the cutoff are purged, and incomplete follow-up is excluded; the real dataset was not evaluated chronologically.
* Repository check after implementation — branch and HEAD unchanged; only the acknowledged reports and new v2 files are untracked. The protected report hashes matched their prior values.

## Remaining blockers

1. Recover authoritative questionnaire, event adjudication, target-generation code, and follow-up/censoring records for the local data.
2. Verify each feature's source timestamp and data-availability contract; the p168-named fields exceed 72-hour history, while the additional lag convention remains unresolved.
3. Run the implemented chronological evaluation only after verified outcome intervals and sufficient follow-up are available.
4. Define alert consequences and acceptable false-alert burden before enabling a threshold or intervention.
5. Connect a backend feature builder to the exact 22-input schema and pinned runtime. The current UI does not provide those inputs; no partner-owned integration files were changed.
6. Conduct external/prospective validation and appropriate subgroup assessment before any clinical or deployment claim.
