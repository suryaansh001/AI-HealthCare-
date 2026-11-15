# ANCHOR AI HEALTHCARE — PHASE 1 DATASET AUDIT REPORT
**Document ID:** `ANCHOR-PHASE1-DATASET-AUDIT`  
**Execution Timestamp:** 2026-11-15T01:30:35+05:30  
**Audit Lead:** Autonomous Machine Learning Research Pipeline  
**Working Branch:** `mohit`  

---

## 1. Executive Summary

This report establishes the verified, independently calculated statistical baseline of the primary feature datasets and raw survey records in the Anchor repository. Every statistic was computed directly from disk using executable Python routines on the complete 274,179-row dataset.

---

## 2. Dataset Dimensions & Participant Demographics

### Primary Feature Matrix (`dataset/features_0lag_v3.parquet` / `.csv`)
- **Total Rows / Time Windows:** $274,179$
- **Total Columns:** $286$ (284 original features + `y` binary target + `hours_since_start`)
- **Total Unique Participants (`subid`):** $151$
- **Observation Window Spacing:** Hourly longitudinal time series ($99.62\%$ of consecutive observations are spaced exactly 1 hour apart).
- **Date Range:** `2017-03-03 06:00:00+00:00` to `2019-12-04 04:00:00+00:00` (975 calendar days).
- **Duplicate Records:** **Zero.** There are no duplicate `[subid, dttm_label]` timestamps.
- **Constant Features:** **Zero.** No zero-variance numeric features exist.

### Participant Window Contribution Distribution
| Statistic | Windows per Participant |
|:---|---:|
| **Minimum** | 472 |
| **25th Percentile** | 1,675 |
| **Median** | 2,010 |
| **Mean** | 1,815.8 |
| **75th Percentile** | 2,055 |
| **Maximum** | 2,136 |

---

## 3. Class Balance & Participant Clustering

### Global Target Balance
- **Positive Lapse Windows (`y = 1`):** $21,107$ ($7.698\%$)
- **Negative Abstinence Windows (`y = 0`):** $253,072$ ($92.302\%$)
- **Class Imbalance Ratio:** Approximately $1 : 12$

### Participant-Level Target Disparity
- **Participants with $\ge 1$ Positive Window:** **84 / 151 (55.63%)**
- **Participants with 0 Positive Windows:** **67 / 151 (44.37%)**
- **Positive Windows Distribution among Lapsers:**
  - **Median:** $165.0$ windows
  - **Mean:** $251.3$ windows
  - **Maximum:** **1,193 windows** for a single hyper-frequent lapsing participant.
  - **Top Participant Share:** The single heaviest lapsing participant accounts for **5.65% of all positive labels** in the entire study.

> **Critical Methodological Takeaway:** Over 44% of the cohort remained completely abstinent throughout monitoring. A small minority of chronic lapsers accounts for the bulk of positive observations. Cross-validation MUST strictly group by participant (`StratifiedGroupKFold`) to avoid memorizing chronic drinker profiles.

---

## 4. Missingness Profile

- **Total Missing Cells:** $2,908,812$ out of $78,415,194$ total cells ($3.710\%$).
- **Columns with At Least One Missing Value:** $252$ out of $286$.
- **Distribution of Missingness:**
  - `static` demographics (8 columns): **0.00% missing**.
  - `past_use` lapse counts (10 columns): **0.00% missing**.
  - `compliance` prompt counts (10 columns): **0.00% missing**.
  - `current_raw` recent EMA (9 columns): **0.05% missing** (confined to the first few hours before survey prompt #1).
  - `aggregate_raw` & `aggregate_change` (234 columns): **4.53% missing** (concentrated in rolling median features `rmedian` and `dmedian` when fewer than required responses were recorded).

---

## 5. Raw Survey Ingestion Baseline (`dataset/completedatasetwithALLcsv/ema.csv`)

- **Total Survey Rows:** $45,460$
- **Total Columns:** $19$
- **Total Participants in Survey Log:** $169$ (18 enrolled participants were screened or dropped prior to hourly feature generation).
- **Self-Reported Drinking Events (`ema_1 = 'Yes'`):** $1,437$ ($3.161\%$ of all completed surveys).
- **Participants Reporting Drinking:** $104$ out of $169$ ($61.54\%$).
- **Survey Cadence:** 4 prompts daily (1 morning survey, 3 random day/evening surveys). Morning surveys exclusively contain Items 8, 9, 10 (measuring planned risk and self-efficacy).

---

## 6. Audit Conclusion & Reproducibility Verification

1. The data matrices are intact, free of duplicates, and have no missing static or past-use variables.
2. The discrepancy between raw drinking events ($1,437$) and positive prediction windows ($21,107$) is verified: because the prediction target uses a rolling 168-hour (7-day) forward window, each drinking episode marks up to 168 adjacent hourly windows as positive.
3. The dataset is ready for Phase 2: Target Validity and Temporal Design.
