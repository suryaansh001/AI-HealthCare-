# Dataset EDA and statistical importance

This document records the exploratory data analysis and feature-family importance results already produced for the alcohol-lapse forecasting dataset. It is intentionally limited to dataset structure, quality checks, and predictive importance. It does not describe the web UI or deployment.

## Dataset overview

Primary feature file:

- [`dataset/features_0lag_v3.csv`](./dataset/features_0lag_v3.csv)
- Parquet copy: [`dataset/features_0lag_v3.parquet`](./dataset/features_0lag_v3.parquet)
- Feature metadata: [`dataset/column_dictionary.csv`](./dataset/column_dictionary.csv)

| Statistic | Value |
|---|---:|
| Rows / labelled time windows | 274,179 |
| Total columns | 284 |
| Participants (`subid`) | 151 |
| Date range | 2017-03-03 to 2019-12-04 |
| Positive lapse windows | 21,107 (7.70%) |
| Negative/non-lapse windows | 253,072 (92.30%) |
| Mean windows per participant | 1,815.8 |
| Median windows per participant | 2,010 |
| Participant window range | 472–2,136 |
| Total missing cells | 2,908,812 |
| Overall cell-level missingness | 3.74% |
| Columns with at least one missing value | 252 / 284 |

The label is `lapse`, with values `yes` and `no`. The modelling target `y` is the binary encoding of that label.

## Feature composition

The 284 columns contain metadata, static participant characteristics, and engineered temporal signals:

| Feature family | Count | Description |
|---|---:|---|
| Metadata | 4 | `label_num`, `subid`, `dttm_label`, `lapse` |
| Static | 8 | Weekday and participant demographics |
| Past-use | 10 | Recent and changed lapse-use rates/counts |
| Compliance | 10 | Recent and changed EMA response/compliance rates |
| Current raw | 9 | Current raw EMA response summaries |
| Current change | 9 | Current deviation/change EMA summaries |
| Aggregate raw | 117 | Raw rolling summaries over longer windows |
| Aggregate change | 117 | Change/deviation rolling summaries over longer windows |
| **Total** | **284** |  |

The temporal feature design uses:

- Look-back periods: 0, 12, 24, 48, 72, and 168 hours
- Feature lag: 0 hours in this feature file
- EMA item identifiers: 2 through 10
- Raw (`r`) and change/deviation (`d`) variants
- Rolling statistics including recent, minimum, maximum, median, and rate/count summaries

The `change` features describe deviation from a person’s own recent state. They should be interpreted as within-person change signals, not as fixed population-level measurements.

## EDA findings

### Target balance

The dataset is imbalanced: only 7.70% of windows are labelled as lapse windows. Accuracy is therefore not an appropriate primary metric. The saved evaluations use AUROC, AUPRC, recall among the top 10% highest-risk windows, Brier score, and within-person AUROC.

Because the positive class is relatively uncommon, AUPRC is especially important: it measures how well the model identifies positive windows relative to the observed positive rate.

### Participant structure

Rows are repeated windows from the same 151 participants rather than independent observations. The number of windows varies substantially by participant. Random row-level splitting would risk leakage between train and test data, so the modelling runs use participant-grouped cross-validation.

### Missingness

There are 2,908,812 missing cells, corresponding to 3.74% of all cells. Missing values are concentrated in rolling median response features; the most-missing columns have approximately 12.53% missingness.

Missingness is not evidence that a signal is unimportant. It may reflect unavailable EMA history or insufficient observations for a rolling statistic. Missing-value handling should be performed inside the modelling pipeline and evaluated without using future labels.

### Static distributions

Selected categorical distributions:

- Sex: Male 141,088 rows; Female 133,091 rows
- Education: College or more 146,981; Some college 100,535; High school or less 26,663
- Employment: Employed 178,737; Unemployed 95,442
- Marital status: Never Married 116,918; Other 97,563; Married 59,698

These are row-level counts. They are not participant-level counts, because each participant contributes multiple time windows.

## Demographic comparisons of lapse windows

The following comparisons use the demographic columns and the binary `lapse` label. They describe **lapse-window prevalence**, not total alcohol consumption, amount consumed, or a clinical diagnosis. The dataset does not support the claim that a demographic group is inherently “more prone” without additional adjustment and validation.

### Observed lapse-window rates

| Demographic | Group | Participants | Windows | Lapse windows | Lapse-window rate |
|---|---|---:|---:|---:|---:|
| Education | College or more | 81 | 146,981 | 11,772 | 8.01% |
| Education | Some college | 55 | 100,535 | 7,088 | 7.05% |
| Education | High school or less | 15 | 26,663 | 2,247 | 8.43% |
| Sex | Female | 74 | 133,091 | 11,876 | 8.92% |
| Sex | Male | 77 | 141,088 | 9,231 | 6.54% |
| Employment | Employed | 98 | 178,737 | 14,081 | 7.88% |
| Employment | Unemployed | 53 | 95,442 | 7,026 | 7.36% |
| Marital status | Never Married | 67 | 116,918 | 11,725 | 10.03% |
| Marital status | Married | 32 | 59,698 | 4,130 | 6.92% |
| Marital status | Other | 52 | 97,563 | 5,252 | 5.38% |

At the row level, the largest observed differences are:

- **Sex:** Female windows have a higher lapse-window rate than male windows (8.92% vs 6.54%).
- **Marital status:** Never-married windows have a higher rate than married windows (10.03% vs 6.92%) and the `Other` category (5.38%).
- **Education:** The rates are close; `Some college` is lowest at 7.05%, while `High school or less` is 8.43%. The high-school-or-less group contains only 15 participants.
- **Employment:** Employed and unemployed rates are close (7.88% vs 7.36%).

### Participant-level “ever had a lapse” check

To reduce the effect of participants contributing different numbers of windows, each participant was counted once and classified as `ever_lapse = 1` if any of their windows had `lapse = yes`.

| Demographic | Group | Participants | Participants with ≥1 lapse | Percentage |
|---|---|---:|---:|---:|
| Education | College or more | 81 | 47 | 58.0% |
| Education | Some college | 55 | 28 | 50.9% |
| Education | High school or less | 15 | 9 | 60.0% |
| Sex | Female | 74 | 48 | 64.9% |
| Sex | Male | 77 | 36 | 46.8% |
| Employment | Employed | 98 | 52 | 53.1% |
| Employment | Unemployed | 53 | 32 | 60.4% |
| Marital status | Never Married | 67 | 43 | 64.2% |
| Marital status | Married | 32 | 14 | 43.8% |
| Marital status | Other | 52 | 27 | 51.9% |

The participant-level pattern is more cautious:

- Female participants were more likely than male participants to have at least one lapse window (64.9% vs 46.8%). A simple participant-level chi-square test gives `p = 0.038`, but this is unadjusted and should not be treated as a causal or clinical result.
- Education, employment, and marital-status differences in the participant-level ever-lapse outcome are not statistically conclusive in this small sample (`p = 0.670`, `p = 0.489`, and `p = 0.129`, respectively).
- The participant-level test for education is especially underpowered for `High school or less` because that group contains only 15 participants.

### Important interpretation

These are unadjusted descriptive comparisons. They do not control for age, participant history, observation length, baseline use, time period, or other correlated factors. Row-level significance tests are also anti-conservative when repeated windows from the same participant are treated as independent. The participant-level summaries are safer for a first descriptive view, but they still have limited power.

Therefore, the defensible summary is:

- The data show higher observed lapse-window prevalence among female and never-married groups.
- The data do **not** establish that education or employment status causes higher alcohol consumption or relapse.
- This analysis measures lapse labels only. It does not directly measure whether a person consumes more alcohol overall.
- A stronger analysis would use participant-clustered regression or mixed-effects models, adjust for baseline use and observation time, report uncertainty intervals, and validate findings in an independent sample.

## Predictive feature-family importance

The saved grouped cross-validation results show a clear pattern:

1. **Past-use features are highly informative.**  
   The `B_past_use` model reaches AUROC 0.888 and AUPRC 0.624 in [`dataset/exact_baseline_summary.csv`](./dataset/exact_baseline_summary.csv), compared with AUROC 0.499 for static features alone.

2. **Current raw signals add predictive information.**  
   Adding current raw signals to static plus past-use features improves AUROC from 0.886 to 0.908 in the exact-baseline experiment (`C_static+past_use` to `D_C+current_raw`). Recall in the top 10% risk windows increases from 0.597 to 0.629.

3. **Change/deviation features are useful but are not the strongest standalone family.**  
   The change-only model reaches AUROC 0.905 and AUPRC 0.654 in the exact-baseline experiment. Its within-person AUROC is 0.609, showing useful temporal ranking within participants, while the raw/current combination provides stronger overall discrimination.

4. **Adding the full feature set produces only a small gain over focused families.**  
   `H_all+compliance` reaches AUROC 0.909 and AUPRC 0.657 in the exact-baseline results. This is only marginally different from the focused `D_C+current_raw` result (AUROC 0.908, AUPRC 0.651), so more features do not automatically mean materially better performance.

5. **Static demographics alone are close to chance for between-window prediction.**  
   `A_static` has AUROC 0.499 in the exact-baseline experiment. This indicates that the temporal behavioral and EMA-derived signals carry most of the predictive information in these runs.

### Exact-baseline experiment

Metrics below are from [`dataset/exact_baseline_summary.csv`](./dataset/exact_baseline_summary.csv). Participant-grouped out-of-fold predictions were used; confidence intervals are participant-bootstrap intervals from the saved run.

| Feature set | Features | AUROC | AUPRC | Within-person AUROC | Recall at top 10% | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Static only | 21 | 0.499 | 0.072 | 0.548 | 0.052 | 0.077 |
| Past use only | 10 | 0.888 | 0.624 | 0.553 | 0.601 | 0.043 |
| Static + past use | 31 | 0.886 | 0.622 | 0.581 | 0.597 | 0.043 |
| + current raw | 40 | 0.908 | 0.651 | 0.636 | 0.629 | 0.042 |
| + raw aggregate state | 157 | 0.909 | 0.647 | 0.628 | 0.618 | 0.042 |
| Change only | 157 | 0.905 | 0.654 | 0.609 | 0.620 | 0.041 |
| Full raw + change | 283 | 0.908 | 0.658 | 0.623 | 0.624 | 0.041 |
| All + compliance | 293 | 0.909 | 0.657 | 0.618 | 0.626 | 0.041 |

The model feature counts are larger than the 284 source columns in some rows because categorical static variables are one-hot encoded and metadata/target columns are excluded from the predictor matrix.

### Robustness note

The repository also contains a separate ablation output in [`dataset/ablation_results_summary.csv`](./dataset/ablation_results_summary.csv). Its results are lower than the exact-baseline table because it uses a different robustness configuration. For example, its `D_C+current_raw` row reports AUROC 0.737 and AUPRC 0.072.

This difference is important: the two files should not be combined into one estimate. When reporting a result, always name the experiment file and configuration that produced it.

## Interpretation limits

- These are predictive associations, not causal effects.
- AUROC and AUPRC describe discrimination in the evaluated sample; they do not establish clinical validity.
- The feature file uses lag 0, so production use would require an explicit review of what information is available at prediction time.
- The model results are grouped by participant, but performance may still vary across people and time periods.
- A high model score should be treated as a prompt for supportive follow-up, not as a diagnosis or certainty that a lapse will occur.

## Source analysis

- Dataset and feature definitions: [`dataset/column_dictionary.csv`](./dataset/column_dictionary.csv)
- Main feature file: [`dataset/features_0lag_v3.csv`](./dataset/features_0lag_v3.csv)
- Ablation implementation: [`run_ablation2.py`](./run_ablation2.py)
- Exact-baseline metrics: [`dataset/exact_baseline_summary.csv`](./dataset/exact_baseline_summary.csv)
- Robustness/ablation metrics: [`dataset/ablation_results_summary.csv`](./dataset/ablation_results_summary.csv)
