    # ANCHOR Master Report: Critical Review and Evidence Verification

    **Review date:** 2026-11-15  
    **Repository:** `AI-HealthCare-`  
    **Review mode:** Read-only examination of Git history, source, available datasets, OOF artifacts, and the serialized model  
    **Reviewed report:** `ANCHOR_MASTER_AUDIT_AND_EVALUATION.md`  
    **Review branch at start:** `anchor-research-correction`  
    **HEAD at start:** `1daa744959e02b94a3914e7ae7b30415aa8a0053`

    ## Executive conclusion

    The master report contains useful exploratory results, but it overstates what the repository proves. Most critically, the repository does not contain executable source code that creates the `lapse` target. The target's exact event definition, time boundaries, lag alignment, missing-event handling, and censoring policy are therefore **NOT VERIFIED**. The report's “24-hour prospective” interpretation depends on that unverified target definition.

    The saved 24-hour OOF artifact is loadable and its principal ranking and probability metrics reproduce. However, the logistic model is **not calibrated**: the training and serialized pipelines contain plain `LogisticRegression`, and the saved artifact contains no calibration wrapper. The threshold is chosen and evaluated on the same OOF predictions. The personalized-forecasting claim is also too strong: 0.581 is a mean per-person AUROC on participants held out from a pooled model, not chronological within-person training and future prediction.

    Deployment readiness is not established. The 22-column schema is not supplied by the current UI, the dataset's item meanings are not established by its CSV, a 72-hour warm-up cannot support a full 168-hour lookback, no backend implementation exists in the repository, and the saved model fails a direct inference attempt with the available scikit-learn 1.9.1 runtime because it was serialized with scikit-learn 1.7.2.

    No training, serialization, data edits, branch switching, or Git-history operations were performed during this review. The only intended repository addition is this review report.

    ## 1. Repository and Git integrity

    ### Observed state

    | Item | Evidence |
    |---|---|
    | Current branch | `anchor-research-correction` |
    | HEAD | `1daa744959e02b94a3914e7ae7b30415aa8a0053` |
    | Initial working-tree status | Clean: `## anchor-research-correction` |
    | `mohit` ref | `ee22c86bbfb0c7df67b68d6bf48c9b17c23d00a0` |
    | `master` / `origin/master` ref | `1c81f68a286924812fe467dbc92538ef3012997a` |
    | `main` ref | No `main` branch was listed |
    | Research/report commit | `2d14eba Complete Anchor ML research and model development` |

    `2d14eba` is an ancestor of both the current correction branch and the `mohit` ref. `mohit` is an ancestor of the current correction branch; the correction branch has later commits. The research commit added the master report, research scripts, model artifact, and research reports. The current master ref remains at the earlier `1c81f68` and does not contain those added files.

    The report's statement that it ran on `mohit` is consistent with the historical location of the research commit, but a commit records no checked-out branch name. Git history alone cannot prove which branch was checked out when the report was authored. The report's branch statement is therefore **not independently authenticated**. The report's claim of a clean working tree at that earlier time also cannot be established from the present tree.

    No branch was switched and no commit/ref was changed during this review. No changes were made to `mohit` or `master` during the review. The files already on the `mohit` ancestry predate this review.

    ### Commands used

    Read-only Git checks included:

    ```text
    git -C <repo> status --short --branch
    git -C <repo> rev-parse HEAD
    git -C <repo> branch -avv
    git -C <repo> log --oneline --decorate --graph -20
    git -C <repo> diff --name-status master..anchor-research-correction
    git -C <repo> log --all --pretty=format: --name-only -- '*.py'
    git -C <repo> lfs status
    ```

    The tracked Python-source history contains feature/audit/pipeline scripts and `target_temporal_analysis.py`, but no target-generation script. The staged/pending Git LFS status was empty.

    ## 2. Target provenance: critical unresolved issue

    ### What source code actually does

    `research/preprocessing/target_temporal_analysis.py:25-26` reads an existing Parquet feature matrix and derives `y` by converting its existing `lapse` column to 0/1. It does not create `lapse`. Lines 51-60 then compare that existing label with already-engineered `p12`, `p24`, and `p168` features. This is analysis of a pre-existing label, not evidence of its construction.

    `dataset/ema.csv` contains raw columns including `dttm_obs`, `subid`, and `ema_1`. `research/preprocessing/dataset_audit.py:84-85` counts `ema_1 == "Yes"` reports, but does not construct the hourly label. `dataset/completedatasetwithALLcsv/ema_results.md` explicitly notes that the CSV alone does not establish the exact item meanings or whether a reported drink is a distinct clinical lapse event.

    ### Provenance findings

    | Requested detail | Finding |
    |---|---|
    | Raw event field | `ema_1` is a raw field with Yes/No values; whether “Yes” is the precise goal-inconsistent event rule is not established by executable target-construction code. |
    | Event timestamp | `dttm_obs` exists in raw EMA. The rule mapping that timestamp to the prediction/label grid is absent. |
    | Future-window construction | NOT VERIFIED. The repository has lagged matrices and narrative descriptions, not the original implementation. |
    | Inclusive/exclusive endpoints | NOT VERIFIED. |
    | Prediction timestamp | NOT VERIFIED from target-generation code. |
    | Lag alignment | NOT VERIFIED from target-generation code. Matrix and feature names encode lag labels but do not define the target transformation. |
    | Missing EMA/event handling | NOT VERIFIED. |
    | Incomplete follow-up / censoring | NOT VERIFIED. No target code establishes whether a no-event window with incomplete follow-up is unknown, censored, or negative. |
    | Participant follow-up cutoff | NOT VERIFIED. Dataset date ranges do not establish per-person censoring behavior. |

    **Conclusion: target provenance is NOT VERIFIED.** The 168-hour forward-window description and the EMA event interpretation must not be presented as source-code-verified facts until the upstream label-generation implementation and its data dictionary are recovered.

    ## 3. Lag-24 early-warning and temporal leakage claims

    The schema names its horizon `24_hours_ahead_of_next_7_days` (`research/models/feature_schema.json:4`), and the data file is named `features_24lag_v3.csv`. The compact schema includes, for example, `p168.l24.*` and `ema_2.p0.l24.rrecent_response`. These labels are evidence of the intended naming convention, not proof of the timestamps used to compute each feature or the outcome window.

    The source needed to establish an exact prediction time, latest feature timestamp, and outcome interval is missing. Therefore:

    * The claim that the model has exactly 24 hours of lead time is **NOT VERIFIED**.
    * Whether any predictor overlaps the target outcome period is **NOT VERIFIED**.
    * Whether the outcome is future drinking, ongoing drinking, or continuation of a previously started episode is **NOT VERIFIED** from the original target code.
    * The serialized OOF file does not contain timestamps or fold IDs, so the exact sample-to-fold assignment cannot be audited from that artifact alone.

    The training source uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` (`research/experiments/train_and_validate.py:88`) and assigns each fold's predictions only to its test indices (`:115`). This code is designed to keep participants in separate folds. It does **not** implement chronological evaluation or outcome-window purging. Under the report's stated 168-hour outcome horizon and 6-hour thinning, adjacent retained outcome windows would overlap substantially if those stated intervals are correct; that overlap makes rows dependent and reduces the effective independent sample size. It does not, by itself, put the same participant in train and test for the grouped split.

    For incomplete follow-up, the training script maps every existing `lapse` value directly to a binary label. It cannot correct censoring if the upstream matrix incorrectly encoded incomplete windows as negative.

    ## 4. Independent recomputation of available OOF metrics

    The repository artifact `research/validation/oof_predictions_24lag.parquet` loaded successfully. It contains 45,206 rows, 151 unique participants, 3,479 positives, and 41,727 negatives (prevalence 0.0769588). Predictions are float32. The recomputation used the saved OOF probabilities and the standard definitions:

    * AUROC: ROC AUC of `y` and probability.
    * AUPRC: average precision.
    * Brier: mean squared error between binary `y` and probability.
    * Log loss: binary cross-entropy, clipping probabilities to `[1e-15, 1 - 1e-15]`.
    * Within-person AUROC: mean of each participant's AUROC among eligible participants.

    | OOF candidate | AUROC | Average precision | Brier | Log loss | Within-person AUROC, >=1 positive and >=1 negative | Within-person AUROC, >=3 positive and >=3 negative |
    |---|---:|---:|---:|---:|---:|---:|
    | `HGB_Full_G` | 0.885212 | 0.462900 | 0.052849 | 0.185691 | 0.557366 (84 people) | 0.557366 (84 people) |
    | `HGB_Compact_K` | 0.874941 | 0.448431 | 0.054278 | 0.191354 | 0.539851 (84 people) | 0.539851 (84 people) |
    | `Logit_Calibrated_K` | 0.871258 | 0.474847 | 0.053648 | 0.192792 | 0.581210 (84 people) | 0.581210 (84 people) |
    | `RandomForest_Compact_K` | 0.884291 | 0.462353 | 0.053320 | 0.186369 | 0.499972 (84 people) | 0.499972 (84 people) |

    The >=1/1 and >=3/3 eligibility results coincide because the artifact has the same 84 participants eligible under both cutoffs; the remaining 67 participants have no positive or no negative examples for a within-person AUC.

    ### Participant-level uncertainty

    For the selected logit and full HGB results, a 250-resample participant-cluster percentile bootstrap (seed 42; participants sampled with replacement, retaining each sampled participant's rows) gave:

    | Candidate | AUROC 95% cluster-bootstrap interval | Average precision 95% interval | Brier 95% interval | Log-loss 95% interval | Mean within-person AUROC 95% interval |
    |---|---|---|---|---|---|
    | `Logit_Calibrated_K` | [0.836281, 0.903015] | [0.340814, 0.577407] | [0.042394, 0.066180] | [0.156671, 0.231129] | [0.536332, 0.622165] |
    | `HGB_Full_G` | [0.854270, 0.911903] | [0.351746, 0.556734] | [0.040962, 0.065828] | [0.148824, 0.225846] | [0.513195, 0.593855] |

    These are exploratory participant-cluster intervals on an OOF artifact, not prospective or external-validation intervals. The 2.5th/97.5th percentiles across the 84 individual participant AUROCs for logit are 0.182971/0.956327 (median 0.595363); those quantiles describe heterogeneity, not a confidence interval for the mean.

    ### Brier score discrepancy (~0.053 vs ~0.18)

    The probability Brier score for `Logit_Calibrated_K` is 0.053648. A Brier score must be computed on probabilities, not on thresholded class labels. Reproducing the hard-class calculation gives:

    | Threshold applied before Brier calculation | Brier on hard 0/1 labels (equal to classification error) |
    |---:|---:|
    | 0.06 | 0.198049 |
    | 0.07 | 0.174534 |
    | 0.08 | 0.155842 |
    | 0.50 | 0.067845 |

    Thus, thresholding probabilities before computing Brier yields approximately 0.18 at threshold 0.07, which plausibly explains a later value around 0.18. However, the repository contains no verification record that identifies the exact operation behind that later value. **The numerical discrepancy is reproducible as a probability-vs-hard-label metric mix-up, but the provenance of the previously quoted 0.18 is NOT VERIFIED.**

    ## 5. Calibration and threshold selection

    `train_and_validate.py:76` constructs a plain `LogisticRegression(C=0.1, ...)`. It imports `CalibratedClassifierCV` at line 15 but does not instantiate or use it. OOF probabilities come directly from each fold pipeline's `predict_proba` (`:115`). The validation metrics therefore do not evaluate an explicit calibration wrapper.

    `serialize_model.py:52` also constructs plain `LogisticRegression`; `joblib.dump` at line 60 saves the preprocessing pipeline plus that estimator. Direct inspection of the artifact confirmed its step classes are `ColumnTransformer` and `LogisticRegression`, with no calibration step. The model card's “Calibrated L2-Regularized Logistic Regression” description is **incorrect**.

    No separate calibration set or independent calibration evaluation is implemented. OOF predictions are used both to report performance and to select an operating threshold. `train_and_validate.py:134-144` searches thresholds from 0.02 to 0.50 on the same pooled OOF predictions and reports the best Youden J. This makes the selected threshold's reported sensitivity/specificity optimistic as an operating-point estimate.

    The stored validation JSON sets `Logit_Calibrated_K`'s selected threshold to **0.06**. The model card and serialization source hard-code **0.08** (`serialize_model.py:96`, `model_card.json:21`); 0.08 is the stored threshold for `HGB_Compact_K`, not for the logit candidate. The integration guide also applies 0.08. The threshold discrepancy is real and unresolved; 0.08 is not the selected Youden threshold for the chosen logit in the current validation artifact.

    ## 6. Personalization and evaluation design

    The 0.581 result is calculated by scoring each participant's held-out OOF rows and averaging AUROC only for participants with at least three positive and three negative rows (`train_and_validate.py:91-99, 124`). Since 84 participants meet eligibility, it is a descriptive within-person ranking measure of predictions from a pooled model for participants excluded from that fold's training.

    It does **not** establish:

    1. **Unseen-participant generalization:** this is measured by pooled OOF performance over held-out participants, although no external cohort is available.
    2. **Personalized adaptation:** the model is not fitted or updated on each participant's own past data.
    3. **Personalized chronological future prediction:** training and testing are not separated chronologically within known participants.

    The existing 336-hour OOF artifact provides an independently reproducible example of the distinction. For `G_E+change(full)`, the recomputed pooled AUROC is 0.829920 and within-person AUROC is 0.454141 among 79 eligible participants. This supports a poor within-person ranking result for that specific historical run; it does not prove that all two-week models or all future datasets fail.

    ### Valid known-participant chronological evaluation to run before a personalization claim

    Pre-register a participant-wise chronological cutoff and an adequate-history/label-count rule. For each test participant, fit the pooled model only on permitted historical observations, or explicitly define and fit a personalized/partial-pooling model only on that participant's past. Test exclusively on subsequent timestamps. If the intended workflow is personalized for already-enrolled participants, do not allow future observations from a test participant into training.

    For a target interval documented as `[prediction_time + lag, prediction_time + lag + horizon)`, purge training rows for the same participant whose outcome interval reaches or overlaps the first test outcome interval. Under the report's proposed 24-hour lag and 168-hour horizon, that interval would nominally span 192 hours from prediction time, but the exact endpoints must first be verified from the missing target generator. Also address overlap among test outcomes (for example with event-level sampling or dependence-aware uncertainty), report each participant and the distribution of participant metrics, and use participant-level bootstrap intervals. A history requirement must be tied to feature requirements, not selected as an unsupported 72-hour rule.

    ## 7. Warm-up and feature availability

    The 22-column model schema includes ten past-use variables with nominal 12/24/48/72/168-hour periods and `l24` lag, plus four `p0.l24` recent EMA values. The maximum named history period is 168 hours. A 72-hour warm-up cannot provide a full 168-hour history feature. If `l24` means the feature window is shifted back by 24 hours relative to prediction time, the earliest source data could be approximately 192 hours before prediction time; this is conditional because the generator is absent.

    Neither the training pipeline nor the serialized model has an enrollment-time warm-up guard. The 72-hour “calibrating” behavior appears only in the model card/integration-guide prose and pseudocode. Numeric imputation does not make unavailable historical measurements valid. Whether the original data pre-fill counts, treat missing history as zero, or use another convention is not established by the source present.

    **Finding:** 72 hours is not a verified or sufficient warm-up for the named p168 inputs. Exact minimum history and missing-history behavior are NOT VERIFIED; under the nominal feature names, at least 168 hours of relevant history is required, and potentially 24 additional hours for the lag.

    ## 8. Frontend/backend/model compatibility

    The actual 22 inputs are listed in `research/models/feature_schema.json`. The current form in `index.html:147` collects mood/steadiness, resting heart rate, and sleep. `app.js:112-114` uses those values in a client-side arithmetic score; there is no call to an inference API. `research/reports/BACKEND_INTEGRATION_GUIDE.md` is a proposed contract, not an implemented backend. It also maps EMA item meanings that are not established by `ema.csv` or its local EDA documentation.

    | Model input | Expected source / construction | Available from current app now? | Required history / status |
    |---|---|---|---|
    | `label_day` | Calendar-day feature | No explicit model payload | Derivable from a verified prediction timestamp; exact convention unknown |
    | `demo_age` | Intake demographic | No | Static intake required |
    | `demo_sex` | Intake demographic | No | Static intake required |
    | `demo_educ` | Intake demographic | No | Static intake required |
    | `demo_income` | Intake demographic | No | Static intake required |
    | `demo_marital` | Intake demographic | No | Static intake required |
    | `demo_race` | Intake demographic | No | Static intake required |
    | `demo_employ` | Intake demographic | No | Static intake required |
    | `p12.l24.rratecount.count.lapse` | Past lapse count/rate feature | No | Nominal 12h period plus l24 shift; event rule unavailable |
    | `p12.l24.dratecount.count.lapse` | Past-use change/deviation feature | No | Same; exact reference baseline unavailable |
    | `p24.l24.rratecount.count.lapse` | Past lapse count/rate feature | No | Nominal 24h period plus l24 shift; event rule unavailable |
    | `p24.l24.dratecount.count.lapse` | Past-use change/deviation feature | No | Same; exact reference baseline unavailable |
    | `p48.l24.rratecount.count.lapse` | Past lapse count/rate feature | No | Nominal 48h period plus l24 shift; event rule unavailable |
    | `p48.l24.dratecount.count.lapse` | Past-use change/deviation feature | No | Same; exact reference baseline unavailable |
    | `p72.l24.rratecount.count.lapse` | Past lapse count/rate feature | No | Nominal 72h period plus l24 shift; event rule unavailable |
    | `p72.l24.dratecount.count.lapse` | Past-use change/deviation feature | No | Same; exact reference baseline unavailable |
    | `p168.l24.rratecount.count.lapse` | Past lapse count/rate feature | No | Nominal 168h period plus l24 shift; 72h cannot supply full history |
    | `p168.l24.dratecount.count.lapse` | Past-use change/deviation feature | No | Same; exact reference baseline unavailable |
    | `ema_2.p0.l24.rrecent_response` | Most-recent raw `ema_2` response at the encoded lag | No | Must collect the item at the correct time; item meaning/scale not verified locally |
    | `ema_6.p0.l24.rrecent_response` | Most-recent raw `ema_6` response at the encoded lag | No | Same |
    | `ema_7.p0.l24.rrecent_response` | Most-recent raw `ema_7` response at the encoded lag | No | Same |
    | `ema_10.p0.l24.rrecent_response` | Most-recent raw `ema_10` response at the encoded lag | No | Same |

    The model has no wearable inputs. The interface's heart rate and sleep values are simulated or client-entered UI fields and are not model features. No current frontend form supplies the demographics, the lapse-history aggregates, or the four model EMA columns with the model's training-time encoding. **The frontend cannot currently invoke this model with a valid, complete payload.**

    ## 9. Saved model, schema, versions, and inference

    Direct inspection of `research/models/anchor_lapse_model_v1.joblib` found:

    * Pipeline steps: `ColumnTransformer` then `LogisticRegression`; no calibrator.
    * Model feature count/order: 22 features; the fitted preprocessing `feature_names_in_` matches `feature_schema.json`.
    * Input handling: the pipeline classifies object/string columns as categorical and the remaining model columns as numeric. No public request schema or input validator is included.
    * Training script: final estimator is refit on the complete thinned 24-lag dataset (`serialize_model.py:34-54`); this is not an independent validation run.
    * Serialization verification: the script reloads the file and predicts on two synthetic examples (`:112-145`), then checks only that the “high” score exceeds the “low” score and both are within [0,1]. It does not compare pre-save and post-load predictions or check repeated-call equality. The reported synthetic scores are not clinical validation.
    * Runtime compatibility: deserialization emitted scikit-learn warnings identifying the stored estimator version as **1.7.2** while the available runtime is **1.9.1**. A direct `predict_proba` call then failed with `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`. The project-level `.venv` contains only `pip`; no repository dependency lock/requirements file pins a compatible scikit-learn version.

    The artifact's feature order is verified, but successful inference is not portable across the available environment. The serialized model is not demonstrated to be deployment-ready. Restore/use the exact tested environment and add a supported, pinned inference environment before further deployment testing; do not rely on a version range such as `1.7+`.

    ## 10. Major claim audit

    | Report claim | Classification | Evidence-based assessment |
    |---|---|---|
    | `P(lapse=1 | p24 > 0) = 99.985%` | **PARTIALLY VERIFIED** | Stored 0-lag analysis metric is 0.9998536 (99.985%). The label definition and temporal relationship are not verified because target-generation code is missing. |
    | Lag 24 is a genuine one-day prospective early-warning task | **NOT VERIFIED** | Lag is encoded in filenames/schema/features, but exact target boundaries and feature timestamps are unavailable. |
    | Lag 24 is optimal | **NOT VERIFIED** | No pre-specified utility criterion, chronological test, or independent horizon-selection evaluation establishes optimality. |
    | Compact model retains >98% of full-model signal | **PARTIALLY VERIFIED** | The ratio 0.871/0.885 is about 98.4% for pooled AUROC in the current OOF comparison. An AUROC ratio is not a validated “signal retention” or non-inferiority measure; other report sections compare different model runs/configurations. |
    | Change features yield zero gain | **VERIFIED INCORRECT** as a literal zero | The lag-24 exploratory report gives a G-minus-E AUROC difference of +0.002 with interval [-0.002, +0.008]. This is compatible with no clear added AUROC, not proof of exactly zero gain. |
    | Within-person AUROC is 0.581 | **PARTIALLY VERIFIED** | Recomputed as 0.581210 for 84 OOF participants. It is an average within-person ranking statistic from a pooled model, not personalized chronological forecasting. |
    | Two-week individual timing prediction fails (0.454) | **PARTIALLY VERIFIED** | Recomputed for the historical `G_E+change(full)` 336-hour OOF run as 0.454141 among 79 eligible participants. This is evidence for that run only, not a universal impossibility claim. |
    | A 72-hour warm-up is sufficient | **VERIFIED INCORRECT** for the named feature set | Model features include p168 histories (with l24 lag). No code validates the 72-hour rule; 72 hours cannot supply a full 168-hour window. |
    | The model is calibrated | **VERIFIED INCORRECT** | Training and artifact contain plain LogisticRegression; no calibration estimator is used. |
    | Threshold 0.08 is the selected Youden threshold for the chosen model | **VERIFIED INCORRECT** | Current validation JSON selects 0.06 for logit. 0.08 is hard-coded in model-card/guide and matches a different candidate's stored threshold. |
    | Production model readiness / reliable inference | **VERIFIED INCORRECT** | No serving backend or valid UI-to-model payload exists; the artifact fails inference under available scikit-learn 1.9.1; no dependency lockfile is present. |
    | Deterministic reload was demonstrated | **VERIFIED INCORRECT** as stated | The script predicts after reload and asserts ordering/range, but does not compare with pre-save predictions or repeat predictions. |
    | Clinical utility / clinically meaningful prediction | **NOT VERIFIED** | No prospective trial, intervention evaluation, clinical workflow study, external validation, decision-curve analysis, or appropriate clinical utility evidence is present. |
    | Wearable physiology is used by the model | **VERIFIED INCORRECT** | The 22-feature schema contains no HR, sleep, or steps. The UI displays those as simulated/client fields. |
    | Raw item labels establish craving/stress/risky-context/self-efficacy semantics | **NOT VERIFIED** | Local EMA documentation says exact item meanings require the questionnaire/codebook; the integration guide's semantic mappings are proposals, not established by available source. |

    ## 11. Recommended research corrections (not implemented)

    1. Recover and version the authoritative label-generation code, source questionnaire/codebook, and censoring rules. Add tests for event and interval boundaries and distinguish positive, observed negative, and censored/unknown windows.
    2. Publish a machine-readable temporal contract defining prediction timestamp, feature cutoff for each feature family, target interval, timezone, and inclusive/exclusive boundaries.
    3. Rebuild evaluation around a pre-registered chronological split for known participants, with outcome-window purging/embargo and event/participant-aware uncertainty. Keep grouped unseen-participant validation as a separate estimand.
    4. Re-evaluate calibration on independent data, if probabilities are needed. Keep threshold selection separate from final evaluation; choose thresholds using explicit support/alert-cost requirements rather than reusing the reported OOF set.
    5. Define each of the 22 input features operationally, including source collection, scale, history sufficiency, missingness, and a valid cold-start policy. Do not equate missing history with zero.
    6. Pin and test an inference environment compatible with the serialized model, or deliberately rebuild/revalidate an artifact in an agreed supported environment. Add a request-schema contract and end-to-end integration tests before claiming service readiness.
    7. Reword reports to describe exploratory performance on this cohort and to avoid clinical-validity, utility, and personalized-forecasting claims unsupported by the current evidence.

    ## 12. Scope and verification limitations

    The review did not regenerate any feature matrix, train a model, or modify any dataset/model/source file. Target provenance could not be established from available repository history. The `0.18` Brier source cannot be tied to a specific historical command; hard-label thresholding reproduces a nearby value but is only a plausible explanation. Cluster-bootstrap intervals use 250 resamples and should be treated as approximate exploratory intervals. The metrics depend on upstream labels whose construction remains unverified.
