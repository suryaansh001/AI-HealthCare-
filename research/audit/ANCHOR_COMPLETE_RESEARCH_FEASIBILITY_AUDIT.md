# ANCHOR Complete Research Feasibility Audit

**Review date:** 2026-11-15  
**Repository:** `AI-HealthCare-`  
**Branch / HEAD:** `anchor-research-correction` / `1daa744959e02b94a3914e7ae7b30415aa8a0053`  
**Mode:** Read-only examination; one new audit report created  
**Purpose:** Independently assess data/event lineage, outcome validity, temporal claims, metrics, model, and implementation feasibility.

## Executive conclusion

The current evidence does **not** establish Anchor as a personalized or clinically useful early-warning system. The central reason is target provenance: the local feature matrices contain a `lapse` column, but no local executable code creates it. The likely original study's publicly available executable label script creates *participant-day* labels from a curated `lapses_day.csv`; it does not create a rolling 168-hour future target or derive an event directly from raw `ema_1`. The missing intermediate event table and imperfect cohort correspondence prevent proving that this particular script generated the local files. Thus the local target's exact source and temporal meaning remain **NOT VERIFIED**; the future-week interpretation is contradicted by the located upstream code.

The grouped out-of-fold (OOF) scores are reproducible as scores against the supplied labels. They are retrospective, participant-held-out results, not evidence that the specified future event can be forecast at a verified lead time. A within-person AUROC of 0.581 is a pooled model's ranking score within held-out participants; it does not demonstrate personalized chronological forecasting. The named feature history includes 168-hour summaries, so a 72-hour warm-up cannot support the complete named feature set. The current UI does not provide the 22 model inputs and has no inference API.

There is a serializable model artifact. It loads and produces a synthetic output under scikit-learn 1.7.2, and its feature order matches the schema. However, the estimator is not calibrated, the threshold selection is not independent, required input-generation rules are undocumented, and no backend/UI integration exists. This is not deployment or clinical validation.

**Recommendation:** Do not advertise or activate the existing artifact as a 24-hour prospective, personalized, clinically useful risk model. First recover and version the exact local target/event pipeline and questionnaire, define a verifiable prediction-time contract, then run a pre-registered chronological evaluation with censoring and overlapping outcomes handled explicitly. Keep the existing UI explicitly marked as a simulation.

## 1. Repository integrity and investigation scope

### Verified state

| Check | Result |
|---|---|
| Repository root | `C:/Users/Mohit sharma/Desktop/7th sem/ai in heathcare/project/AI-HealthCare-` |
| Current branch | `anchor-research-correction` |
| HEAD | `1daa744959e02b94a3914e7ae7b30415aa8a0053` |
| Branch refs observed | `mohit` at `ee22c86`; `master` and `origin/master` at `1c81f68`; no `main` ref listed |
| Initial working-tree status | Only the two acknowledged, untracked audit reports |
| Requested output before creation | Absent |
| Protected reports | Present; hashes recorded below |

History showed `2d14eba` (“Complete Anchor ML research and model development”) on the ancestry leading through `mohit` to the correction branch; `mohit` is at `ee22c86`, while the correction branch has later commits `eb58b7c` and `1daa744`. The research commit is therefore present in the history of the branch; that alone cannot prove which branch was checked out when the report was authored. No branch was switched and no commit/ref was changed during this audit.

Protected report SHA-256 values before report creation:

* `research/audit/ANCHOR_MASTER_REPORT_CRITICAL_REVIEW.md`: `7B53072BA001ED32F786055BE649CAFF31D63CB0A29BA1E5DBB81C6BE1E86771`
* `research/audit/ANCHOR_TARGET_PROVENANCE_INVESTIGATION.md`: `0FD87D5BE484533568F2625F726FC54C2EFB45E0AF617726AA55E2DEFFBAA2CB`

### Scope limitations

No model was trained, no source/data/model was rewritten, no package was installed, and no Git history was modified. Metric recomputation used existing OOF predictions in memory. The two acknowledged reports and all pre-existing project files were treated as read-only. The upstream public code and OSF pages are evidence of the study's published materials, but they do not prove direct file-by-file lineage to the local feature matrices.

## 2. Dataset and event-generation lineage

### What is locally available

The local raw EMA file is [`dataset/ema.csv`](../../dataset/ema.csv) (also present in `dataset/completedatasetwithALLcsv/`). The recorded audit found 45,460 rows, 19 columns, and 169 participants. It contains `subid`, `dttm_obs`, `ema_1`, and related fields. `ema_1` contains 44,023 `No` and 1,437 `Yes` values, with no missing values. These are survey responses, not independently verified drinking episodes. The local EDA explicitly says item semantics require a questionnaire/codebook and cautions that one affirmative response is not necessarily a distinct lapse event ([`ema_results.md`](../../dataset/completedatasetwithALLcsv/ema_results.md)).

The five local engineered CSV matrices are under `dataset/completedatasetwithALLcsv/`. The 0-hour matrix has 274,179 rows, 284 columns and 151 participants; the 24-hour matrix has 270,911 rows and the same 151 participants. The other observed row counts are 264,362 (72h), 251,458 (168h), and 228,420 (336h). The local `lapse` labels agree on shared `(subid, dttm_label)` keys between adjacent lag matrices; this supports a common repeated label but does not establish how it was generated. The 0-hour labels are also consistent within participant-day bins at 04:00 Chicago time. These checks are recorded in the existing [target-provenance investigation](ANCHOR_TARGET_PROVENANCE_INVESTIGATION.md).

### Public upstream materials located

The OSF Data component's public description says it provides raw EMA data and feature sets for three models (week, day, hour); the Methods component says it provides the parent protocol and consent form; the Scripts component describes fitting, selection, evaluation, and training-control code. The preprint is [Wyant et al., “Machine learning models for temporally precise lapse prediction in alcohol use disorder”](https://osf.io/preprints/psyarxiv/cgsf7), DOI `10.31234/osf.io/cgsf7`. These establish broad research lineage, not exact provenance of the local five-lag files.

The executable [`mak_lapse_labels.qmd`](https://github.com/jjcurtin/study_messages/blob/1c09e66fd05bcf9b448b677c59a64b6ba073906e/_mak/messages_archive/mak_lapse_labels.qmd) is the relevant public label script located in the original study materials. It reads `study_dates.csv` and `lapses_day.csv` and writes `lapses.csv`. Those intermediates and the upstream event-table creation process are not present locally. The messages-specific cohort filters also do not establish an exact match to the local 151-participant matrix. Consequently, this is strong evidence about what the located upstream script does, but **not proof that this exact script version produced every local label**.

## 3. Target provenance: what the executable label code says

The local [`target_temporal_analysis.py`](../preprocessing/target_temporal_analysis.py) reads an existing Parquet feature matrix and converts its existing `lapse` values to `y`; it computes runs, autocorrelation, and associations with existing past-use features. It does **not** create `lapse`. Likewise, [`train_and_validate.py`](../experiments/train_and_validate.py) binarizes an existing `lapse` column. Neither is target-generation evidence.

In the located upstream Quarto/R label script:

1. **Daily grid:** It builds participant-day rows from study dates, starting at `start_study + 1 day` and ending at `end_study - 1 day`, then sets the hour to 04:00. It does not explicitly reset seconds.
2. **Event source/time:** It reads `lapse_start_date`, `lapse_start_time`, end fields, `exclude`, and `ema_1_6` from `lapses_day.csv`. It parses a reconstructed `lapse_start`, applies `ema_1_6` as a timezone, and converts the timestamp to America/Chicago. This is not a direct read or transformation of raw `ema_1`.
3. **Positive event:** A record with `exclude == "FALSE"` and a non-missing start is assigned to the participant-day containing that start. The event's end time does not extend the positive label.
4. **Boundary:** `day_end` is coded as `day_start + 1 day - 1 second`; the comparisons are inclusive at both ends. Thus the intended coded interval is 04:00 through 03:59:59, excluding the next 04:00. Any seconds inherited when constructing `day_start` are not explicitly cleared.
5. **Excluded events:** An `exclude == "TRUE"` event marks the day containing its start and the day containing its end as `"exclude"`; those days are removed. Intermediate days of a multi-day event are not explicitly marked by this loop.
6. **Missingness/default:** A comment says a missing start time should be `NA`, but executable code fills it with `20:00`. Missing end date is filled with the day after the start date; missing end time is filled with `02:00`. A row without a parsed start bypasses the valid/excluded loops. Every remaining unlabelled day is assigned `"no lapse"`; there is no explicit unknown/censored day state.
7. **Participant selection:** The source notes dropping participants without follow-up 1 and excludes IDs 104, 269, and 204 for data-quality/lapse-reporting reasons; it also notes hand-editing one start time. This does not resolve the local matrix's exact cohort or event lineage.

| Target question | Evidence-based result |
|---|---|
| Is the local event exactly `ema_1 == Yes`? | **NOT VERIFIED.** The located generator reads curated `lapses_day.csv`, not `ema.csv`; `ema_1_6` is used as timezone metadata. |
| Is the located upstream output a rolling 168-hour future target? | **No.** It labels individual participant-days based on event start time. |
| What exact local timestamp/window does `lapse` describe? | **NOT VERIFIED.** Local label-generation and matrix-export code are missing. |
| Is there a label lag shift in the located generator? | **No.** The script assigns event starts to study days; it contains no future-window shift or lag parameter. |
| How are missing EMA and missed event reports handled locally? | **NOT VERIFIED.** The local target creator and upstream `lapses_day.csv` creation process are unavailable. |
| Are incomplete future windows treated as negative? | There is no future-window logic in the located upstream day-label script. It defaults unassigned days to `"no lapse"`, but cannot be used to quantify local future-window censoring. |
| Is local participant censoring fully known? | **NOT VERIFIED.** Participant start/end dates and upstream exclusions exist in public materials, but local matrix construction and observation completeness are not reproducible from available files. |

**Finding:** Do not describe the supplied local `lapse` target as a source-verified rolling future drinking window derived directly from `ema_1`. The precise local target, event definition, lag alignment, and censoring remain **NOT VERIFIED**; the located upstream executable contradicts the rolling-week interpretation.

## 4. Lag-24 prospective validity and dependent windows

`features_24lag_v3.csv`, feature names containing `l24`, and [`feature_schema.json`](../models/feature_schema.json) document a 24-hour *naming convention*. The schema's literal `prediction_horizon` string is `24_hours_ahead_of_next_7_days`. None of these artifacts defines the actual prediction timestamp, latest permitted source timestamp, timezone, or local target interval.

The following are therefore **NOT VERIFIED** for the local data:

* exact prediction timestamp and latest legal feature timestamp;
* the exact outcome start/end and inclusive/exclusive boundaries;
* whether any predictor's source observations overlap the outcome;
* whether lag 24 has exactly 24 hours of prospective lead time;
* whether the target is future onset, same-day event, or episode continuation;
* whether a no-event label has complete follow-up.

The latest training script uses five-fold `StratifiedGroupKFold` by participant, with shuffling, and assigns OOF predictions to held-out participants. Thus one participant is not split between train and test for this evaluation. It does not perform chronological validation or purge temporal outcome windows. If the hypothesized target were a 168-hour window and examples were retained every six hours, adjacent *within-person* outcome windows would overlap by 162 hours. That creates strongly dependent test rows and a smaller effective information base; it does not put the same participant in train and test under the grouped split. The saved primary OOF artifact has no timestamps or fold IDs, so it cannot audit temporal cutoffs or purging.

The original target is not verified, so it is not valid to call Lag 24 “genuine,” “clinically meaningful,” or “clinically validated” early warning.

## 5. Recomputed metrics and uncertainty

### Primary saved OOF artifact

I independently loaded [`oof_predictions_24lag.parquet`](../validation/oof_predictions_24lag.parquet) in memory with pandas/pyarrow and recomputed standard metrics from its saved binary `y` and probabilities. It has 45,206 rows, 151 participants, 3,479 positives, and prevalence `0.07695881`; its columns are `subid`, `y`, and four model probability columns. It contains no timestamps or fold IDs.

| OOF candidate | AUROC | Average precision (AUPRC) | Brier | Log loss | Mean within-person AUROC |
|---|---:|---:|---:|---:|---:|
| `HGB_Full_G` | 0.885212 | 0.462900 | 0.052849 | 0.185691 | 0.557366 |
| `HGB_Compact_K` | 0.874941 | 0.448431 | 0.054278 | 0.191354 | 0.539851 |
| `Logit_Calibrated_K` | 0.871258 | 0.474847 | 0.053648 | 0.192792 | 0.581210 |
| `RandomForest_Compact_K` | 0.884291 | 0.462353 | 0.053320 | 0.186369 | 0.499972 |

Within-person AUROC is the unweighted mean of participant-level AUROCs, requiring at least three positive and three negative rows. Exactly 84 of 151 participants qualify; 67 do not have both classes at the required count. The per-person AUROC distribution for the logit has median `0.595363` and 2.5th/97.5th percentiles `0.182971`/`0.956327`. These percentiles describe between-participant heterogeneity, not a confidence interval for the mean.

For the selected logit, a participant-cluster percentile bootstrap (250 resamples with replacement, seed 42, retaining all rows for each sampled participant) gave:

| Metric | Point estimate | Participant-cluster 95% interval |
|---|---:|---:|
| AUROC | 0.871258 | [0.837098, 0.896441] |
| Average precision | 0.474847 | [0.343963, 0.582144] |
| Brier | 0.053648 | [0.041605, 0.065561] |
| Log loss | 0.192792 | [0.155971, 0.229108] |
| Mean within-person AUROC | 0.581210 | [0.540792, 0.621196] |

For each bootstrap replicate the within-person mean was computed over sampled participants eligible under the same 3-positive/3-negative rule. These are exploratory intervals over saved predictions, not prospective, external-validation, or model-refitting intervals; 250 replicates are coarse.

### Keep the experiment families separate

The repository also contains a distinct multi-repeat feature-family ablation (`updated_24lag_*`) and a different 336-hour run. The 24-hour full-set result in `updated_24lag_summary.csv` is AUROC `0.891410`, AUPRC `0.470819`, within-person AUROC `0.554339`; the 336-hour full-set result is `0.829920`, `0.240283`, and `0.454141` among 79 eligible participants. These are not the same OOF run or selected estimator as the `research/validation` four-candidate artifact. The 24-hour matrix itself has 270,911 rows, not the 270,555 shown in the Stage 2 inventory. Do not mix the separate experiments into one model-performance claim.

### Brier discrepancy

The proper Brier score is the mean squared error between the *probabilities* and binary outcomes: the primary OOF logit value is `0.053648`. Thresholding first makes the same calculation equal to classification error. Recalculation on hard classes gives:

| Probability threshold | Brier on hard labels (classification error) |
|---:|---:|
| 0.06 | 0.198049 |
| 0.07 | 0.174534 |
| 0.08 | 0.155842 |
| 0.50 | 0.067845 |

Thresholding near 0.07 can numerically explain a value around 0.18, but no repository record establishes that this was the source of the later claim. The exact provenance of the `~0.18` value is **NOT VERIFIED**; a probability-vs-hard-label metric mix-up is plausible, not proven.

## 6. Calibration and decision threshold

[`train_and_validate.py`](../experiments/train_and_validate.py) imports `CalibratedClassifierCV` but constructs ordinary `LogisticRegression(C=0.1, ...)` for `Logit_Calibrated_K`. OOF probabilities come directly from `predict_proba`. [`serialize_model.py`](../models/serialize_model.py) likewise fits and serializes a preprocessing pipeline plus ordinary `LogisticRegression`. Direct artifact inspection confirmed steps `ColumnTransformer` and `LogisticRegression`, with **no calibrator**. The “calibrated” estimator name/model-card description is inaccurate.

No separate calibration set, fitted calibration mapping, or independent calibration evaluation is present. A Brier score alone does not establish calibration. Threshold selection searches thresholds on the same pooled OOF scores subsequently reported as validation, so operating-point performance is optimistically selected and not independently evaluated.

The primary [`validation_results.json`](../validation/validation_results.json) selects threshold **0.06** for the logit. The model card, serialization script, and proposed integration guide use **0.08**. The latter is the stored Youden threshold for `HGB_Compact_K`, not for the selected logit in the current validation artifact. Thus **0.08 is not the selected OOF Youden threshold for the chosen logit**, and neither threshold should be treated as an independently validated clinical decision threshold.

## 7. Personalization and a valid known-participant evaluation

The `0.581210` score answers: for each held-out participant with adequate positive and negative rows, how well do probabilities from a *pooled model trained without that participant* rank that participant's positive versus negative rows? It is a descriptive within-person ranking summary. It does not answer whether adapting a model to a known person's history forecasts that person's later outcomes.

Distinguish these estimands:

1. **Unseen-participant generalization:** grouped OOF results estimate performance on held-out participants from this cohort distribution.
2. **Within-person ranking:** compute participant-specific AUCs on their held-out rows, then macro-average eligible people; this is the 0.581 statistic.
3. **Personalized chronological prediction:** train/update using only a known person's past available data (or a pre-defined pooled-plus-personal adaptation), then test on subsequent time. This has not been evaluated.

The 336-hour result (`0.454141` within-person AUROC for one historical full feature-set run, 79 eligible people) is below 0.5 for that particular run. It does not prove that every two-week model or all future datasets must fail.

### Required evaluation before a personalization claim

After defining a defensible target interval and feature cutoff:

* Pre-register one or more chronological cutoffs per participant. Fit pooled parameters only on permitted historical training participants/periods; for personalized adaptation, use only that test participant's pre-cutoff history.
* At each test origin, use only information actually available then. Do not allow labels whose full outcome interval ends after the training cutoff.
* Purge/embargo training origins whose outcome intervals overlap test outcomes. If the target is later verified as a 168-hour interval, adjacent six-hour origins are not independent; report the exact purge interval dictated by the target definition.
* Set an adequate-history rule from actual feature-generation requirements (at least the full named p168 lookback, subject to the 24-hour lag convention) and a pre-specified minimum number of observed positive and negative outcomes. Report excluded participants and reasons.
* Report participant-level denominators, per-person AUROC/AUPRC where defined, pooled and macro-averaged measures, calibration, uncertainty resampled by participant, and performance at multiple calendar cutoffs. Keep this known-user estimand separate from unseen-participant grouped validation.

## 8. Feature history, availability, and warm-up

The 22-input schema contains a nominal p168 history and `l24` on all lagged features. A 72-hour warm-up cannot supply the entire nominal 168-hour history. If `l24` means the feature window ends 24 hours before the prediction origin, complete p168 history may require 192 hours from enrollment; this additional 24 hours is conditional because the feature-generation code is missing. The name alone cannot establish the latest source observation.

The training pipeline uses a median imputer for numeric columns and a constant `"missing"` imputer plus one-hot encoding for categorical columns. That is a model preprocessing behavior, not a validated policy for unavailable history. The upstream feature builder's early-history rules, zero-filling, missingness, and any burn-in are not available. Do not equate an imputed value with measured absence of drinking or a complete personal baseline.

### Exact model-input mapping

All 22 inputs below are required by the serialized schema. “Unavailable” means the current UI does not provide a value with the model's training-time identity, type, and temporal definition.

| Model input | Source / construction required | Available from current UI? | Minimum nominal history | Status |
|---|---|---:|---|---|
| `label_day` | Calendar-day category at prediction origin | No explicit payload | None; timestamp convention required | Not wired |
| `demo_age` | Intake demographic | No | None | Missing |
| `demo_sex` | Intake demographic | No | None | Missing |
| `demo_educ` | Intake demographic | No | None | Missing |
| `demo_income` | Intake demographic | No | None | Missing |
| `demo_marital` | Intake demographic | No | None | Missing |
| `demo_race` | Intake demographic | No | None | Missing |
| `demo_employ` | Intake demographic | No | None | Missing |
| `p12.l24.rratecount.count.lapse` | Past lapse count over nominal p12 at encoded l24 | No | p12 + lag convention | Missing; source event rule unresolved |
| `p12.l24.dratecount.count.lapse` | Past-use change/deviation for p12 at l24 | No | p12 + lag and baseline convention | Missing; baseline unresolved |
| `p24.l24.rratecount.count.lapse` | Past lapse count over nominal p24 at encoded l24 | No | p24 + lag convention | Missing; source event rule unresolved |
| `p24.l24.dratecount.count.lapse` | Past-use change/deviation for p24 at l24 | No | p24 + lag and baseline convention | Missing; baseline unresolved |
| `p48.l24.rratecount.count.lapse` | Past lapse count over nominal p48 at encoded l24 | No | p48 + lag convention | Missing; source event rule unresolved |
| `p48.l24.dratecount.count.lapse` | Past-use change/deviation for p48 at l24 | No | p48 + lag and baseline convention | Missing; baseline unresolved |
| `p72.l24.rratecount.count.lapse` | Past lapse count over nominal p72 at encoded l24 | No | p72 + lag convention | Missing; source event rule unresolved |
| `p72.l24.dratecount.count.lapse` | Past-use change/deviation for p72 at l24 | No | p72 + lag and baseline convention | Missing; baseline unresolved |
| `p168.l24.rratecount.count.lapse` | Past lapse count over nominal p168 at encoded l24 | No | p168 + lag convention; potentially 192h total | Missing; 72h is insufficient |
| `p168.l24.dratecount.count.lapse` | Past-use change/deviation for p168 at l24 | No | p168 + lag and baseline convention; potentially 192h | Missing; baseline unresolved |
| `ema_2.p0.l24.rrecent_response` | Most-recent EMA item 2 response at encoded l24 | No | Observation availability + l24 | Missing; item wording needs codebook |
| `ema_6.p0.l24.rrecent_response` | Most-recent EMA item 6 response at encoded l24 | No | Observation availability + l24 | Missing; item wording needs codebook |
| `ema_7.p0.l24.rrecent_response` | Most-recent EMA item 7 response at encoded l24 | No | Observation availability + l24 | Missing; item wording needs codebook |
| `ema_10.p0.l24.rrecent_response` | Most-recent EMA item 10 response at encoded l24 | No | Morning/item availability + l24 | Missing; item wording needs codebook |

The UI collects a mood slider, sleep hours, and heart rate; the JS computes a heuristic score from those values. It has no call to a backend and no payload containing demographics, past-use features, or these four encoded EMA inputs. The guide's proposed item-to-construct mappings are not verified by the local CSV/codebook. The UI's simulated heart rate/sleep/steps are not model inputs. The frontend cannot currently invoke the model with a valid complete row.

## 9. Model artifact and inference compatibility

The saved [`anchor_lapse_model_v1.joblib`](../models/anchor_lapse_model_v1.joblib) is 7,594 bytes. Under an available scikit-learn 1.7.2 environment it loaded without warnings. Its class is `Pipeline`, with `ColumnTransformer` and `LogisticRegression` steps. The fitted `feature_names_in_` order exactly matches the 22 entries in `feature_schema.json`.

The preprocessing partition observed in the artifact is:

* categorical: `label_day`, `demo_sex`, `demo_educ`, `demo_marital`, `demo_race`, `demo_employ`;
* numeric: `demo_age`, `demo_income`, ten past-use count/change features, and four EMA response values.

A one-row synthetic input built with the fitted categories and numeric zeros produced probability `[0.9939, 0.0061]`; repeated prediction on that same in-memory artifact/input was identical. This demonstrates that a technically valid inference call can run in the matching scikit-learn 1.7.2 environment. It does **not** validate the synthetic user's data, calibrate the probability, test pre- versus post-serialization parity, or establish clinical prediction.

The serialization script fits on all thinned data and reloads the saved artifact before its synthetic high/low assertion. It checks score ordering and probability bounds, not pre/post-save prediction equality. No calibration object is serialized. The repository does not provide a complete pinned inference environment or a backend request/response implementation. A direct load in one matching environment is not proof of portable deployment.

## 10. Audit of major research and engineering claims

The classifications below refer to the claim **as stated**, not whether a related exploratory statistic exists.

| Major claim | Classification | Evidence-based conclusion |
|---|---|---|
| Local `lapse` is a rolling 168-hour future drinking outcome derived directly from `ema_1` | **NOT VERIFIED** | No local target creator exists. The located upstream executable makes daily labels from `lapses_day.csv` and does not read raw `ema_1` or create future rolling windows. |
| `P(lapse=1 \| p24>0) = 99.985%` proves continuation confounding | **PARTIALLY VERIFIED** | Stored 0-lag calculation is 0.9998536. It is an association among existing labels/features, not proof of future continuation because local label semantics are unresolved. |
| Lag 24 is a genuine one-day prospective warning model | **NOT VERIFIED** | Exact target interval, prediction origin, and feature cutoff are missing. |
| Lag 24 is optimal / clinically actionable | **NOT VERIFIED** | No pre-specified utility criterion, chronological evaluation, or independent horizon-selection test establishes optimality or actionability. |
| Compact model retains 98% of full-model signal | **PARTIALLY VERIFIED** | `0.871258 / 0.885212 ≈ 98.4%` for pooled AUROC in this OOF run. A ratio of AUROCs is not non-inferiority or general signal-retention evidence; other runs differ. |
| Change features provide exactly zero gain | **VERIFIED INCORRECT** as a literal zero | The 24h participant-bootstrap contrast is approximately `+0.000` AUROC with CI `[-0.003,+0.004]`, `+0.004` AP with CI `[-0.009,+0.019]`, and `-0.002` within-AUROC with CI `[-0.023,+0.017]`. This is no clear added value in this run, not proof of exact zero. |
| Within-person AUROC is 0.581 | **PARTIALLY VERIFIED** | Recomputed as 0.581210 for 84 eligible participants. It is ranking by a pooled model on held-out participants, not personalized chronological forecasting. |
| Two-week individual forecasting fails | **PARTIALLY VERIFIED** | One 336h full-feature run has within-person AUROC 0.454141 among 79 eligible participants, with pooled AUROC 0.829920. This is specific-run evidence, not a universal impossibility result. |
| A 72-hour warm-up supports all model features | **VERIFIED INCORRECT** for the named schema | The schema includes p168 histories; 72 hours cannot supply the complete named lookback. Exact added lag/history rule remains unverified. |
| The logistic model is calibrated | **VERIFIED INCORRECT** | Training and artifact contain ordinary `LogisticRegression`, no calibrator or independent calibration evaluation. |
| 0.08 is the selected logit threshold | **VERIFIED INCORRECT** | Stored validation selects 0.06 for logit; 0.08 belongs to HGB compact and is used in the model card/guide. |
| The saved model is production-ready | **VERIFIED INCORRECT** | No inference service exists; UI inputs do not satisfy the 22-feature contract; feature/history generation and dependency lock are absent. Artifact loading in a matching environment does not cure these blockers. |
| The model offers clinical utility / clinically validated predictions | **NOT VERIFIED** | No prospective/external validation, decision-curve or utility analysis, intervention trial, or clinical workflow study is provided. |
| Wearable physiology is used by the fitted model | **VERIFIED INCORRECT** | The schema has no heart-rate, sleep, or step inputs; those are UI simulation/client fields. |
| All EMA feature meanings are verified as craving/stress/risky context/self-efficacy | **NOT VERIFIED** | Local EDA requires the instrument/codebook; the guide's mapping is a proposal. |
| All-zero warm-up inputs correctly represent a new participant | **NOT VERIFIED** | No validated cold-start/missing-history policy exists; training median imputation is not a semantic history rule. |
| Reload test proves deterministic serialized prediction | **PARTIALLY VERIFIED** | The script obtains predictions after reload and asserts ordering/range; it does not compare the original in-memory model with the reloaded model. Independent repeated calls on one loaded artifact were equal, but that is a narrower test. |

## 11. Scientific feasibility and recommended next phase

### What is supportable now

* The supplied feature matrices and OOF artifacts support reproducible retrospective metrics against their stored labels.
* The feature matrices are EMA/demographic, not wearable-sensor datasets.
* The 0-lag data show a strong empirical association between existing past-use counts and existing labels (`p24 > 0`: 99.985%; `p12 > 0`: 100%). This is useful as a warning about label autocorrelation/repetition, but it does not by itself establish the meaning or prospective nature of the target.
* The selected logit OOF ranking metrics and Brier/log-loss values can be stated with the caveat that labels are unverified and samples are clustered/dependent.
* The model artifact can technically produce a score in a matching runtime when supplied correctly typed features.

### Prerequisites before any early-warning efficacy claim

1. Obtain and preserve the authoritative questionnaire, event/adjudication rules, exact `lapses_day.csv`/`study_dates.csv` inputs, and exact target-generation/export version for these local matrices. Reconcile participant IDs and timestamps end-to-end.
2. Create a versioned target specification: event definition, observation source, timezone, prediction origin, feature cutoff, future interval, endpoint convention, follow-up completeness, missing-event handling, and censoring. Represent censored/unknown windows separately from observed negatives.
3. Verify feature-generation semantics and availability for every schema field; specify training-time category/scale encodings and full history requirements. Do not turn unavailable history into a zero-risk signal.
4. Pre-register chronological known-participant testing and a separate unseen-participant estimand. Purge overlapping outcome windows, require labels fully observed before a training cutoff, report person-level performance and cluster intervals, and repeat across cutoffs.
5. If probabilities or alerts are intended, fit/evaluate calibration and choose thresholds on data separate from final evaluation, based on explicit consequences and alert capacity.
6. Make the UI payload, server API, feature builder, artifact, and pinned runtime one tested contract. Keep the current interface labeled as simulated until that contract and an end-to-end test exist.
7. Only after these steps, assess external validation, fairness, safety, clinical workflow, and prospective utility. Avoid claiming clinical validation or utility from retrospective cross-validation.

## 12. Evidence and reproducibility record

### Key local sources

* Target analysis consumes an existing label: [`target_temporal_analysis.py`](../preprocessing/target_temporal_analysis.py), especially the read/`y` conversion and subsequent run/feature association calculations.
* Current grouped OOF training and threshold selection: [`train_and_validate.py`](../experiments/train_and_validate.py), including the 24h CSV load, stride 6, `StratifiedGroupKFold`, candidate estimators, OOF predictions, and Youden search.
* Fold-safe preprocessing and feature-set construction: [`pipeline.py`](../preprocessing/pipeline.py). It does not reconstruct upstream source features.
* Serialized schema/card/artifact: [`feature_schema.json`](../models/feature_schema.json), [`model_card.json`](../models/model_card.json), [`serialize_model.py`](../models/serialize_model.py), and [`anchor_lapse_model_v1.joblib`](../models/anchor_lapse_model_v1.joblib).
* OOF probabilities and stored metrics: [`oof_predictions_24lag.parquet`](../validation/oof_predictions_24lag.parquet) and [`validation_results.json`](../validation/validation_results.json).
* Historical ablation evidence: `updated_24lag_summary.csv`, `updated_24lag_contrasts.csv`, and `updated_336lag_summary.csv` at repository root; `dataset/h3_quiet_withpast_summary.csv` is under `dataset/`. Results from these are not the same experiment as the primary four-model OOF artifact.
* Current frontend: [`index.html`](../../index.html), [`app.js`](../../app.js), and [`styles.css`](../../styles.css). The integration guide is a proposal, not a running backend.
* Raw item data limitations: [`ema.csv`](../../dataset/ema.csv) and [`ema_results.md`](../../dataset/completedatasetwithALLcsv/ema_results.md).

### Commands/checks performed

Read-only Git checks:

```text
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --branch
git branch -avv
git --no-pager log --oneline --decorate --graph -18
git --no-pager log --oneline --all -- <target/training/model source paths>
```

Searches and inspections used targeted `rg` over target/training/serialization/UI files; direct reads of the local reports, schema, model card, JSON metrics, CSV summaries, EMA EDA and feature dictionary; and the public upstream `mak_lapse_labels.qmd`, OSF component descriptions, and preprint metadata.

Metric recomputation (in memory only) used the saved OOF parquet:

```python
y = df["y"].to_numpy()
p = df[model_name].to_numpy()
roc_auc_score(y, p)
average_precision_score(y, p)
brier_score_loss(y, p)
log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))
```

Within-person AUROC was calculated separately for each `subid` with at least three positives and three negatives, then macro-averaged. The participant bootstrap resampled 151 participant IDs with replacement 250 times and retained each selected participant's full OOF rows; the interval endpoints are the empirical 2.5th and 97.5th percentiles.

Model inspection used the existing `.joblib` in an available scikit-learn 1.7.2 environment and did not retrain or rewrite it. One synthetic schema-typed row was scored twice to check repeat-call equality; the synthetic score is not a clinical example or validation.

### Final protected-state check

This report is the only file created for the comprehensive investigation. The two protected reports remain unmodified. Final branch, HEAD, status, and SHA-256 values were checked after writing; see the completion message for the resulting values.
