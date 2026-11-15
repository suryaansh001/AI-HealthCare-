# ANCHOR v3 Personalization Experiments

## Outcome

The v3 pipeline and grouped experiments ran against the existing fixed v2 locked
test split. The v2 model and test set were not changed, no v3 model was serialized,
and no frontend or backend files were modified.

The past-EMA hybrid improved several metrics against the stored test labels,
including within-participant AUROC. This is an exploratory result, not evidence
of prospective alcohol-lapse forecasting: the target's timing, feature
availability, and follow-up remain **NOT VERIFIED**. No v3 candidate is selected
for deployment. V2 remains the fixed benchmark.

## Design and reproducibility

- Source matrix: `dataset/completedatasetwithALLcsv/features_24lag_v3.csv`.
- Input schema and fixed model: `../models/feature_schema_v2.json` and
  `../models/anchor_lapse_model_v2.joblib`.
- Fixed metric record: `../validation/model_comparison_v2.json`.
- Data loading and grouped split logic reuse `../anchor_v2/data.py` and
  `../anchor_v2/validation.py`.
- Stride-six matrix: 45,206 rows from 151 participants. The locked test is the
  same 6,068 rows from 20 participants, with 485 positive stored labels
  (prevalence 7.99%).
- All candidates were compared on the same locked test. Candidate thresholds
  were selected using the separate v2 threshold-selection participants; the
  saved v2 threshold was retained for v2. The constant and label-persistence
  diagnostics use a fixed 0.5 threshold.
- Candidate selection used five-fold `StratifiedGroupKFold` OOF predictions
  within the v2 development participants. The selected OOF candidate was the
  compact pooled RF (mean rank 2.25); the full personalized candidate tied its
  mean rank but used more features. V3 exploratory RFs used 40 trees to keep the
  added cross-validation run tractable. This setting does not alter the
  serialized v2 model.
- Participant-cluster bootstrap intervals use 1,000 replicates. Paired
  candidate-v2 deltas use the same test participants. Intervals are
  unadjusted for the multiple exploratory comparisons and should be treated
  cautiously with only 20 test participants (12 eligible for within-person
  AUC).
- Runtime: Python 3.10.19, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2,
  joblib 1.5.3, Matplotlib 3.10.9. Matplotlib was installed in the local
  experiment environment after the plotting import failed; no repository
  dependency manifest was changed.

From the repository root, reproduce with:

```powershell
python -m research.anchor_v3.run --seed 42 --bootstrap-replicates 1000
python -m unittest research.anchor_v3.tests.test_v3 -v
```

The experiment refuses to overwrite its named result and figure files.

## Methods implemented

The v3 features are added in `features.py`. They include:

- Per-person rolling means, standard deviations, deviations, and observed
  counts for four EMA response features.
- Per-person historical missingness summaries and current missingness flags.
- Nominal p24-versus-p168 lapse-rate contrasts.

EMA summaries use only the preceding 24 retained (stride-six) rows for that
participant, exclude the current row, and require at least six prior
non-missing values for rolling means and standard deviations. The window is a
**sampled-row count**, not a verified number of hours. The input timestamps and
source feature windows have not been established as available at prediction
time. Thus these tests establish implementation order, not prospective
leakage safety.

The experiment compares the fixed v2 RF with a development-prevalence
constant, a p24-history-presence baseline, compact logistic regression,
historical-count-only logistic regression, EMA-only RF, pooled compact RF,
EMA-personal-history hybrid RF, p24/p168-rate augmented RF, and a full hybrid
RF. A previous stored-label score is reported only as an autocorrelation
diagnostic; its target availability and overlap are unknown, so it is not a
forecast baseline.

Participant IDs are excluded from model features and output tables. Participant
diagnostics use pseudonymous tokens. Separate participant-specific models
were not fit: the fixed unseen-participant split has no training labels for
locked-test participants. A valid known-participant chronological experiment
is blocked pending target intervals, endpoint convention, follow-up, and
feature-timing evidence.

## Fixed v2 benchmark and locked-test results

AUROC, AUPRC, Brier, and log loss are threshold-independent. Threshold metrics
are shown at the exploratory operating points described above. False alerts
are per 100 test rows, not per person-day.

| Model / diagnostic | AUROC | AUPRC | Brier | Log loss | Within-person AUROC (eligible/20) | Precision | Recall | Specificity | False alerts / 100 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed v2 RF | 0.8431 | 0.3522 | 0.06099 | 0.21417 | 0.4855 (12/20) | 0.244 | 0.775 | 0.791 | 19.25 |
| Constant prevalence | 0.5000 | 0.0799 | 0.07363 | 0.27914 | 0.5000 (12/20) | 0.000 | 0.000 | 1.000 | 0.00 |
| History-presence prevalence | 0.6300 | 0.1811 | 0.06680 | 0.25200 | 0.5390 (12/20) | 0.423 | 0.295 | 0.965 | 3.21 |
| Previous stored-label diagnostic | 0.9392 | 0.7962 | 0.01840 | 0.62189 | 0.9044 (12/20) | 0.886 | 0.882 | 0.990 | 0.91 |
| Pooled compact RF (40 trees) | 0.8436 | 0.3538 | 0.06109 | 0.21422 | 0.4798 (12/20) | 0.191 | 0.843 | 0.689 | 28.59 |
| Compact logistic regression | 0.8442 | 0.3638 | 0.06205 | 0.21930 | 0.5727 (12/20) | 0.290 | 0.724 | 0.846 | 14.14 |
| Historical-count-only logistic | 0.8452 | 0.3481 | 0.06196 | 0.21862 | 0.4136 (12/20) | 0.214 | 0.864 | 0.724 | 25.43 |
| EMA-only RF (40 trees) | 0.8093 | 0.2919 | 0.06563 | 0.23420 | 0.6216 (12/20) | 0.240 | 0.503 | 0.862 | 12.72 |
| EMA personal-history hybrid RF | 0.8710 | 0.3877 | 0.05943 | 0.20331 | 0.5242 (12/20) | 0.212 | 0.874 | 0.718 | 25.92 |
| p24/p168-rate augmented RF | 0.8433 | 0.3575 | 0.06099 | 0.21528 | 0.4950 (12/20) | 0.259 | 0.769 | 0.809 | 17.60 |
| Full personal-history hybrid RF | 0.8639 | 0.3891 | 0.05966 | 0.20701 | 0.5022 (12/20) | 0.277 | 0.742 | 0.831 | 15.51 |

The previous-label diagnostic's high ranking score is not a successful model
result. It directly uses the prior stored target, has poor log loss from hard
scores, and may exploit target-window autocorrelation or overlap. It must not be
used to claim prospective performance.

All requested threshold metrics, including sensitivity, F1, alert rate,
confusion counts, and each model's participant-bootstrap intervals, are in
`results/comparison_metrics_v3.json`.

### Paired participant-bootstrap deltas versus v2

Intervals below are 95% participant-cluster bootstrap intervals for
candidate-minus-v2 differences. Negative Brier/log-loss differences favor the
candidate.

| Candidate | AUROC delta | AUPRC delta | Brier delta | Log-loss delta | Within-person AUROC delta |
|---|---:|---:|---:|---:|---:|
| EMA personal-history hybrid RF | [0.0162, 0.0496] | [0.0070, 0.0622] | [-0.00322, -0.00034] | [-0.01896, -0.00457] | [0.00029, 0.08089] |
| Full personal-history hybrid RF | [0.0076, 0.0358] | [-0.00195, 0.07575] | [-0.00336, 0.00016] | [-0.01578, -0.00061] | [-0.02072, 0.05256] |
| EMA-only RF | [-0.0892, 0.0242] | [-0.1276, 0.0373] | [0.00159, 0.00831] | [0.00436, 0.03832] | [0.01902, 0.24784] |

The EMA hybrid improved stored-label pooled and within-person point metrics;
its paired intervals narrowly exclude zero for the five metrics above. The
within-person point estimate rose from 0.4855 to 0.5242, with a higher
participant AUROC for 10 of the 12 eligible participants. This is an
exploratory fixed-test result after multiple comparisons, not a validated
personalized forecasting result. The full hybrid's within-person interval
includes zero. EMA-only has higher within-person discrimination but worse
pooled discrimination and probability scores.

At the threshold selected outside the test set, the EMA hybrid's recall was
0.874, but precision was 0.212 and it generated 25.9 false alerts per 100 rows,
compared with 19.2 for v2. Threshold-dependent results do not establish an
acceptable alert burden or clinical utility.

## Participant and feature diagnostics

- Test participant sizes ranged from 216 to 348 rows. Stored-label prevalence
  ranged from 0 to 0.495. Only 12 of 20 participants had at least three
  positives and three negatives and qualified for within-person AUROC.
- For v2, participant mean prediction variance accounted for 62.7% of the
  decomposed prediction variance. Mean predicted probability and participant
  stored-label prevalence had Spearman correlation 0.907. Centering scores by
  participant mean reduced AUROC to 0.523. These descriptive results suggest
  strong between-participant structure; they do not prove why the model
  predicts or establish target leakage.
- V2 RF impurity importance assigned 76.9% to historical-count features.
  `p168` lapse count was the largest single feature (0.159), followed by p48
  (0.098), p72 (0.089), and p72 change (0.086). Importance is associational,
  not causal evidence or proof of leakage.
- In the full hybrid RF, the added `v3_` features summed to 21.4% impurity
  importance. The largest were `v3_ema_10_prior24_mean` (0.050),
  `v3_nominal_lapse_rate_24_minus_168` (0.046), and
  `v3_nominal_lapse_rate_ratio_24_to_168` (0.034). The rate-augmented model
  alone was essentially at the v2 benchmark on the locked test; there is no
  clear independent benefit established for those rate contrasts.
- The test history strata contain 120 rows with 0-5 prior sampled rows, 360
  with 6-23, and 5,588 with 24 or more. They are descriptive and not
  time-based warm-up strata. In the 0-5-row stratum, personal rolling summaries
  lack their required six prior observations and are imputed by the fitted
  pipeline.
- Participant-level row counts, prevalence, Brier/AUPRC/AUROC, and false
  positive/negative counts are in `results/participant_metrics_v3.csv`.
  Pseudonymous diagnostics are descriptive; participant denominators are
  small and vary substantially.

## Warm-up, temporal validity, and selection

The code establishes a six-prior-nonmissing-reading minimum for rolling EMA
mean/std calculations and a 24-prior-sampled-row maximum window. It does not
establish that 72 hours is enough, that p168 is an exact 168-hour interval, or
that any feature was measured before a verified prediction time. With
stride-six rows and gaps, row counts cannot be converted to hours. The
prediction/outcome overlap and completeness of follow-up remain unknown.

The grouped fixed-test experiment is not a known-participant chronological
evaluation. No test participant contributes labels to model training, but the
personal-history transformations can use earlier feature rows from that
participant's held-out sequence. This is not a strict cold-start assessment
and requires valid measurement timestamps to be interpreted prospectively.
No outcomes were purged by interval because authoritative intervals are
unavailable.

**Decision:** retain v2 as benchmark; do not save or promote a v3 model. The
EMA hybrid is a candidate for further investigation only after target and
feature timing are verified and it is evaluated with multiple chronological
cutoffs, past-only history, outcome-window purging, complete-follow-up
exclusion, and participant-level reporting.

## Artifacts

- Code and tests: `__init__.py`, `features.py`, `diagnostics.py`, `visualize.py`,
  `run.py`, and `tests/test_v3.py`.
- Metrics and OOF: `results/comparison_metrics_v3.json`,
  `results/development_oof_v3.parquet`,
  `results/personalization_diagnostics_v3.json`,
  `results/participant_metrics_v3.csv`,
  `results/history_strata_v3.csv`, and `results/feature_importance_v3.csv`.
- Figures: `figures/participant_auc_v3.png`,
  `figures/participant_mean_risk_v3.png`,
  `figures/history_amount_metrics_v3.png`,
  `figures/v2_feature_importance_v3.png`, and
  `figures/calibration_diagnostics_v3.png`.
- No v3 model artifact was created.
